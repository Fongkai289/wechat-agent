"""Readable text and media references for raw WeChat message payloads."""

from __future__ import annotations

import functools
import mimetypes
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .cli import rel_display

if TYPE_CHECKING:
    from .web import AppState


V2_IMAGE_MAGIC = b"\x07\x08V2\x08\x07"


try:
    from Crypto.Cipher import AES
except Exception:  # pragma: no cover - optional at runtime
    AES = None


@functools.lru_cache(maxsize=8192)
def find_media_candidates(media_root: Path, table_hash: str, msg_type: str, create_time: int, digest: str) -> tuple[str, ...]:
    if not media_root.exists():
        return tuple()
    month = datetime.fromtimestamp(create_time).strftime("%Y-%m") if create_time else ""
    candidates: list[Path] = []
    if msg_type == "video" and month:
        base = Path("video") / month
        for name in (f"{digest}.mp4", f"{digest}_thumb.jpg", f"{digest}.jpg"):
            candidates.append(base / name)
    if msg_type == "image" and month:
        preferred = Path("attach") / table_hash / month / "Img"
        for name in image_candidate_names(digest):
            candidates.append(preferred / name)
        glob_root = media_root / "attach"
        for match in glob_root.glob(f"*/{month}/Img/{digest}*.dat"):
            candidates.append(match.relative_to(media_root))

    seen: set[str] = set()
    out: list[str] = []
    for rel in candidates:
        key = rel.as_posix()
        if key in seen:
            continue
        seen.add(key)
        if (media_root / rel).exists():
            out.append(key)
    return tuple(out)


def image_candidate_names(digest: str) -> tuple[str, ...]:
    return (
        f"{digest}_h_M.dat",
        f"{digest}_M.dat",
        f"{digest}.dat",
        f"{digest}_t_M.dat",
        f"{digest}_t.dat",
    )


def inspect_media_path(path: Path, state: AppState) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError:
        return {"available": False, "encrypted": False}
    encrypted = data.startswith(V2_IMAGE_MAGIC)
    if encrypted:
        decoded = decrypt_v2_image_data(data, state.image_aes_key, state.image_xor_key)
        mime = sniff_mime(decoded[:64], path.name) if decoded else "application/octet-stream"
        return {"available": bool(decoded and mime != "application/octet-stream"), "encrypted": True, "mime": mime}
    mime = sniff_mime(data[:64], path.name)
    return {"available": mime != "application/octet-stream", "encrypted": False, "mime": mime}


def media_url(rel: str) -> str:
    return "/media?file=" + urllib.parse.quote(rel)


def asset_url(rel: str) -> str:
    return "/asset?file=" + urllib.parse.quote(rel)


def display_content(content: str, msg_type: str) -> str:
    text = content or ""
    if msg_type == "text":
        return text
    voip = summarize_voip_message(text)
    if voip:
        return voip
    if msg_type == "system":
        return summarize_system_message(text)
    if msg_type in {"type_35", "type_42", "type_48", "type_66"}:
        return summarize_structured_message(text, msg_type)
    if msg_type == "voip":
        return "[音视频通话]"
    if msg_type == "image":
        return "[图片]"
    if msg_type == "voice":
        return "[语音]"
    if msg_type == "video":
        return "[视频]"
    if msg_type == "app":
        return summarize_app_message(text)
    if msg_type == "sticker":
        return "[表情包]"
    if parse_message_xml(text) is not None or text.lstrip().startswith("<"):
        return f"[暂不支持的消息 · {msg_type}]"
    return text or f"[{msg_type}]"


class MessageRichTextParser(HTMLParser):
    tags = {"a", "span", "div", "p", "section", "br", "img", "b", "strong", "em", "i", "font", "_wc_custom_link_", "script", "style"}
    blocks = {"div", "p", "section", "br"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        if self.hidden:
            return
        if tag not in self.tags:
            self.parts.append(self.get_starttag_text())
        elif tag in self.blocks:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
            return
        if not self.hidden:
            if tag not in self.tags:
                self.parts.append(f"</{tag}>")
            elif tag in self.blocks:
                self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def readable_message_text(text: str) -> str:
    if not re.search(r"</?(?:a|span|div|p|section|br|img|b|strong|em|i|font|_wc_custom_link_|script|style)(?:\s|/?>)", text, re.I):
        return text.strip()
    parser = MessageRichTextParser()
    # HTMLParser requires tag names to start with a letter.
    parser.feed(re.sub(r"(</?)_wc_custom_link_(?=[\s/>])", r"\1a", text))
    parser.close()
    return re.sub(r"\n[ \t\n]+", "\n", "".join(parser.parts)).strip()


def summarize_structured_message(text: str, msg_type: str) -> str:
    labels = {"type_35": "邮件通知", "type_42": "联系人名片", "type_48": "位置", "type_66": "企业微信名片"}
    label = labels[msg_type]
    root = parse_message_xml(text)
    if root is None:
        return f"[{label} · 内容暂无法解析]"
    details = []
    if msg_type == "type_35":
        details = [first_text(root, "pushmail/content/subject"), first_text(root, "pushmail/content/sender"), first_text(root, "pushmail/content/digest")]
    elif msg_type in {"type_42", "type_66"}:
        details = [root.get("nickname") or root.get("username") or "", root.get("alias") or "", root.get("openimdesc") or ""]
    elif msg_type == "type_48":
        location = root.find("location")
        if location is not None:
            details = [location.get("poiname") or "", location.get("label") or ""]
            if not any(details) and location.get("x") and location.get("y"):
                details = [f"{location.get('x')}, {location.get('y')}"]
    parts = list(dict.fromkeys(readable_message_text(part) for part in details if part))
    return f"[{label}]" + ("\n" + "\n".join(parts) if parts else "")


def summarize_system_message(text: str) -> str:
    body = re.sub(r"^\s*[^\s:]+@chatroom:\s*", "", text).strip()
    root = parse_message_xml(body)
    if root is None:
        return "[系统消息]" if body.startswith("<sysmsg") else readable_message_text(body) or "[系统消息]"
    if root.tag != "sysmsg":
        return readable_message_text(body) if root.tag.lower() in MessageRichTextParser.tags else "[系统消息]"
    content = root.find("./sysmsgtemplate/content_template")
    if content is not None:
        plain = find_text(content, "plain")
        if plain:
            return readable_message_text(plain)
        template = find_text(content, "template")
        values: dict[str, str] = {}
        for link in content.findall("./link_list/link"):
            name = link.get("name") or ""
            members = [
                first_text(member, "nickname", "username")
                for member in link.findall("./memberlist/member")
            ]
            separator = link.findtext("separator")
            values[name] = (separator if separator is not None else "、").join(filter(None, members)) or first_text(link, "plain", "title")
        if template == "$username$ invited $names$ to the group chat":
            template = "$username$ 邀请 $names$ 加入了群聊"
        if template:
            # Replace once so placeholder-like text in a nickname stays literal.
            return re.sub(r"\$([^$]+)\$", lambda match: values.get(match[1]) or "某位成员", template)
    notice = first_text(root, "./revokemsg/replacemsg", "./revokemsg/content", "./delchatroommember/plain", "./delchatroommember/text", "./content", "./plain")
    if notice:
        return readable_message_text(notice)
    if root.get("type") == "mmchatroomtopmsg":
        name = find_text(root, "./mmchatroomtopmsg/nickname")
        return f"{name} 更新了群置顶消息" if name else "群置顶消息已更新"
    return "[系统消息]"


VOIP_STATUS_LABELS = {
    "busy": "对方忙线",
    "cancel": "已取消",
    "canceled": "已取消",
    "cancelled": "已取消",
    "declined": "已拒绝",
    "missed": "未接听",
    "reject": "已拒绝",
    "rejected": "已拒绝",
    "timeout": "未接听",
}


def summarize_voip_message(text: str) -> str:
    root = parse_message_xml(text)
    if root is None or root.tag != "voipmsg":
        return ""
    bubble = root.find("VoIPBubbleMsg")
    if bubble is None:
        return "[音视频通话]"

    message = find_text(bubble, "msg")
    msg_type = find_text(bubble, "msg_type")
    duration = parse_optional_int(find_text(bubble, "duration")) or 0

    kind = "音视频通话"
    if duration > 0:
        return f"[{kind}] 通话时长 {format_duration(duration)}"

    label = VOIP_STATUS_LABELS.get(message.strip().casefold())
    if label:
        return f"[{kind}] {label}"
    if msg_type == "100" and message:
        return f"[{kind}] {message}"
    return f"[{kind}]"


def format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s" if sec else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def strip_group_sender_prefix(content: str, sender_username: str) -> str:
    if not content or not sender_username:
        return content
    prefix = f"{sender_username}:"
    if not content.startswith(prefix):
        return content
    body = content[len(prefix) :]
    if body.startswith("\r\n"):
        return body[2:]
    if body.startswith("\n"):
        return body[1:]
    return body.lstrip()


def summarize_app_message(text: str) -> str:
    app = parse_app_message(text)
    if not app:
        return "[应用消息 · 内容暂无法解析]" if "<" in text else compact_long_text(text, "[应用消息]")
    title = app.get("title") or ""
    desc = app.get("description") or ""
    url = app.get("url") or ""
    parts = []
    label = app.get("label") or ""
    if label:
        parts.append(f"[{label}]")
    if title:
        parts.append(title)
    if desc:
        parts.append(desc)
    if not parts and url:
        parts.append(url)
    return "\n".join(parts) if parts else "[链接]"


APP_TYPE_LABELS = {
    "5": "外部链接",
    "6": "文件",
    "19": "聊天记录",
    "33": "小程序",
    "36": "小程序",
    "44": "小程序",
    "51": "视频号",
    "57": "引用",
    "2000": "转账",
}


def parse_app_message(text: str) -> dict[str, Any] | None:
    root = parse_message_xml(text)
    if root is None:
        return None
    appmsg = root if root.tag == "appmsg" else root.find("appmsg")
    if appmsg is None:
        return None
    app_type = find_text(appmsg, "type")
    title = readable_message_text(find_text(appmsg, "title"))
    desc = readable_message_text(find_text(appmsg, "des"))
    url = find_text(appmsg, "url")
    app_name = first_text(root, "appinfo/appname", "appinfo/appname_en") or appmsg.get("appid") or ""
    label = APP_TYPE_LABELS.get(app_type, "应用消息" if app_type else "链接")
    domain = urllib.parse.urlparse(url).netloc if url else ""
    thumb_md5 = first_text(appmsg, "appattach/cdnthumbmd5", "md5")
    return {
        "kind": "app",
        "type": app_type,
        "label": label,
        "title": title or url or label,
        "description": desc,
        "url": url,
        "domain": domain,
        "app_name": app_name,
        "thumb_md5": thumb_md5,
    }


RECORD_ITEM_LABELS = {
    "1": "文本",
    "2": "图片",
    "3": "名片",
    "4": "语音",
    "5": "视频",
    "6": "链接",
    "7": "位置",
    "8": "文件",
    "17": "聊天记录",
    "19": "小程序",
    "22": "视频号",
    "23": "视频号直播",
    "29": "音乐",
    "36": "小程序",
    "37": "表情包",
}


RECORD_BINARY_SUBDIRS = {"2": "Img", "4": "A", "5": "V", "8": "F"}


RECORD_ITEM_LIMIT = 200


def parse_forwarded_record(
    state: AppState | None,
    outer_rec: dict[str, Any] | None,
    outer_create_time: int | None,
    text: str,
) -> dict[str, Any] | None:
    root = parse_message_xml(text)
    if root is None:
        return None
    appmsg = root.find("appmsg")
    if appmsg is None or find_text(appmsg, "type") != "19":
        return None
    title = find_text(appmsg, "title") or "聊天记录"
    desc = find_text(appmsg, "des")
    record_node = appmsg.find("recorditem")
    record_text = (record_node.text or "").strip() if record_node is not None else ""
    record_root = parse_message_xml(record_text)
    if record_root is None:
        return {
            "title": title,
            "description": desc,
            "count": 0,
            "items": [],
            "truncated": False,
            "note": "这条聊天记录卡片没有完整 recorditem，可能需要在微信里点开后才会落本地缓存",
        }

    datalist = record_root.find("datalist")
    items = list(datalist.findall("dataitem")) if datalist is not None else []
    declared_count = parse_optional_int(datalist.get("count") if datalist is not None else None)
    count = declared_count if declared_count is not None else len(items)
    record_dir = find_forwarded_record_dir(state, outer_rec, int(outer_create_time or 0), items)
    parsed_items = [
        parse_forwarded_record_item(state, record_dir, idx, item)
        for idx, item in enumerate(items[:RECORD_ITEM_LIMIT])
    ]
    return {
        "title": find_text(record_root, "title") or title,
        "description": find_text(record_root, "desc") or find_text(record_root, "info") or desc,
        "count": count,
        "items": parsed_items,
        "truncated": len(items) > RECORD_ITEM_LIMIT,
        "is_chatroom": find_text(record_root, "isChatRoom") == "1",
    }


def parse_forwarded_record_item(
    state: AppState | None,
    record_dir: Path | None,
    index: int,
    item: ET.Element,
) -> dict[str, Any]:
    datatype = (item.get("datatype") or "").strip()
    label = RECORD_ITEM_LABELS.get(datatype, f"未知类型 {datatype}" if datatype else "未知类型")
    title = first_text(item, "datatitle", "title")
    desc = first_text(item, "datadesc", "desc")
    content = forwarded_record_item_content(datatype, label, title, desc, item)
    media = resolve_forwarded_record_item_media(state, record_dir, index, datatype, item)
    return {
        "type": datatype,
        "label": label,
        "sender": find_text(item, "sourcename"),
        "time": find_text(item, "sourcetime"),
        "content": content,
        "media": media,
        "dataid": item.get("dataid") or "",
        "source_local_id": find_text(item, "srcMsgLocalid"),
        "source_create_time": find_text(item, "srcMsgCreateTime"),
        "md5": first_text(item, "fullmd5", "thumbfullmd5", "md5", "emojiitem/md5"),
    }


def forwarded_record_item_content(datatype: str, label: str, title: str, desc: str, item: ET.Element) -> str:
    if datatype == "1":
        return desc or title or "[文本]"
    if datatype in {"2", "4", "5"}:
        return ""
    if datatype in {"3", "7", "23", "37"}:
        return f"[{label}]"
    if datatype in {"6", "19", "22", "29", "36"}:
        text = title or desc or first_text(item, "weburlitem/title", "finderFeed/desc")
        return f"[{label}] {text}" if text else f"[{label}]"
    if datatype == "8":
        file_name = title or desc or first_text(item, "datafmt")
        return f"[文件] {file_name}" if file_name else "[文件]"
    return desc or title or f"[{label}]"


def find_forwarded_record_dir(
    state: AppState | None,
    outer_rec: dict[str, Any] | None,
    outer_create_time: int,
    items: list[ET.Element],
) -> Path | None:
    if state is None or outer_rec is None or not outer_create_time:
        return None
    table_hash = outer_rec.get("table_hash") or ""
    if not table_hash:
        return None
    month = datetime.fromtimestamp(outer_create_time).strftime("%Y-%m")
    base = state.media_root / "attach" / table_hash / month / "Rec"
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    scored = [(score_forwarded_record_dir(path, items), path) for path in dirs]
    scored.sort(key=lambda pair: (pair[0], pair[1].stat().st_mtime if pair[1].exists() else 0), reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else None


def score_forwarded_record_dir(path: Path, items: list[ET.Element]) -> int:
    score = 0
    for idx, item in enumerate(items):
        datatype = (item.get("datatype") or "").strip()
        if datatype == "2":
            score += score_candidate_file(path / "Img" / str(idx), item, "datasize", 3)
            score += score_candidate_file(path / "Img" / f"{idx}_t", item, "thumbsize", 2)
            for hit in (path / "Img").glob(f"{idx}_*"):
                if hit.is_file():
                    score += score_candidate_file(hit, item, "datasize", 2)
                    score += score_candidate_file(hit, item, "thumbsize", 1)
        elif datatype == "5":
            score += 4 if (path / "V" / f"{idx}.mp4").exists() else 0
            score += score_candidate_file(path / "Img" / f"{idx}_t", item, "thumbsize", 1)
        elif datatype == "4":
            score += 2 if (path / "A" / str(idx)).exists() else 0
        elif datatype == "8":
            score += 2 if (path / "F" / str(idx)).exists() else 0
    return score


def score_candidate_file(path: Path, item: ET.Element, size_field: str, base_score: int) -> int:
    if not path.exists() or not path.is_file():
        return 0
    expected = parse_optional_int(find_text(item, size_field))
    if expected is None:
        return base_score
    try:
        actual = path.stat().st_size
    except OSError:
        return 0
    # V2 images carry a small header around the encoded image payload, so allow slack.
    return base_score + 3 if abs(actual - expected) <= 128 else base_score


def resolve_forwarded_record_item_media(
    state: AppState | None,
    record_dir: Path | None,
    index: int,
    datatype: str,
    item: ET.Element,
) -> dict[str, Any] | None:
    if state is None:
        return None
    if datatype == "2":
        return resolve_forwarded_record_image(state, record_dir, index, item)
    if datatype == "4":
        return resolve_forwarded_record_voice(state, record_dir, index, item)
    if datatype == "5":
        return resolve_forwarded_record_video(state, record_dir, index, item)
    return None


def resolve_forwarded_record_image(
    state: AppState,
    record_dir: Path | None,
    index: int,
    item: ET.Element,
) -> dict[str, Any] | None:
    full = record_dir / "Img" / str(index) if record_dir else None
    thumb = record_dir / "Img" / f"{index}_t" if record_dir else None
    paths = record_image_candidates(record_dir, index) if record_dir else []
    encrypted = False
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        status = inspect_media_path(path, state)
        rel = rel_display(path, state.media_root)
        encrypted = encrypted or status["encrypted"]
        if status["available"]:
            return {
                "kind": "image",
                "available": True,
                "url": media_url(rel),
                "encrypted": status["encrypted"],
                "hash": first_text(item, "fullmd5", "thumbfullmd5"),
            }
    if encrypted or bool((full and full.exists()) or (thumb and thumb.exists())):
        return {
            "kind": "image",
            "available": False,
            "encrypted": True,
            "hash": first_text(item, "fullmd5", "thumbfullmd5"),
            "note": "转发记录里的图片无法解码，请检查 image_aes_key，或在微信里重新点开下载",
        }
    return missing_forwarded_record_media("2", index, item)


def record_image_candidates(record_dir: Path, index: int) -> list[Path]:
    img_dir = record_dir / "Img"
    if not img_dir.exists():
        return []
    direct = [img_dir / str(index), img_dir / f"{index}_t"]
    globbed = sorted(p for p in img_dir.glob(f"{index}_*") if p.is_file())
    seen: set[Path] = set()
    out: list[Path] = []
    for path in [*direct, *globbed]:
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and path.is_file():
            out.append(path)
    return out


def resolve_forwarded_record_voice(
    state: AppState,
    record_dir: Path | None,
    index: int,
    item: ET.Element,
) -> dict[str, Any] | None:
    if record_dir:
        audio_dir = record_dir / "A" / str(index)
        paths = sorted(p for p in audio_dir.glob("*") if p.is_file()) if audio_dir.exists() else []
        if not paths:
            direct = record_dir / "A" / str(index)
            paths = [direct] if direct.exists() and direct.is_file() else []
        if paths:
            path = paths[0]
            rel = rel_display(path, state.media_root)
            return {
                "kind": "record-voice",
                "available": False,
                "url": media_url(rel),
                "size": path.stat().st_size,
                "note": "转发记录里的语音本地文件已找到，但还没有接入播放/转文字",
            }
    return missing_forwarded_record_media("4", index, item)


def resolve_forwarded_record_video(
    state: AppState,
    record_dir: Path | None,
    index: int,
    item: ET.Element,
) -> dict[str, Any] | None:
    if record_dir is None:
        return missing_forwarded_record_media("5", index, item)
    video = record_dir / "V" / f"{index}.mp4"
    thumb = record_dir / "Img" / f"{index}_t"
    if video.exists() and video.is_file():
        rel_video = rel_display(video, state.media_root)
        rel_thumb = rel_display(thumb, state.media_root) if thumb.exists() and thumb.is_file() else ""
        return {
            "kind": "video",
            "available": True,
            "url": media_url(rel_video),
            "thumb_url": media_url(rel_thumb) if rel_thumb else "",
            "hash": first_text(item, "fullmd5", "thumbfullmd5", "dataid"),
        }
    if thumb.exists() and thumb.is_file():
        status = inspect_media_path(thumb, state)
        rel_thumb = rel_display(thumb, state.media_root)
        if status["available"]:
            return {
                "kind": "image",
                "available": True,
                "url": media_url(rel_thumb),
                "encrypted": status["encrypted"],
                "hash": first_text(item, "thumbfullmd5", "dataid"),
            }
    return missing_forwarded_record_media("5", index, item)


def missing_forwarded_record_media(datatype: str, index: int, item: ET.Element) -> dict[str, Any]:
    label = RECORD_ITEM_LABELS.get(datatype, "媒体")
    kind = {"2": "image", "4": "record-voice", "5": "video"}.get(datatype, "file")
    has_remote_ref = bool(first_text(item, "cdndataurl", "cdnthumburl", "cdndatakey", "cdnthumbkey"))
    note = f"转发记录里的{label}未在本地缓存中找到；请在微信里打开这条聊天记录卡片，点击第 {index + 1} 项下载后，再重新解密并刷新"
    if has_remote_ref:
        note += "（XML 里有 CDN 引用，但不是浏览器可直接打开的文件）"
    return {
        "kind": kind,
        "available": False,
        "missing": True,
        "remote": has_remote_ref,
        "hash": first_text(item, "fullmd5", "thumbfullmd5", "dataid"),
        "note": note,
    }


def format_forwarded_record_summary(record: dict[str, Any]) -> str:
    title = record.get("title") or "聊天记录"
    count = int(record.get("count") or len(record.get("items") or []))
    return f"[聊天记录] {title} · {count} 条"


def parse_message_xml(text: str) -> ET.Element | None:
    start = re.search(r"<(?:\?xml\b|(?:sysmsg|voipmsg|msg|recordinfo|appmsg)\b)", text)
    xml_text = (text[start.start():] if start else text).strip("\x00\ufeff \t\r\n")
    if not xml_text:
        return None
    try:
        return ET.fromstring(xml_text)
    except ET.ParseError:
        # Some WeChat payloads put an XML declaration inside <msg> or append
        # sibling VoIP metadata. Preserve CDATA; only repair that framing.
        repaired = re.sub(r"^(<msg\b[^>]*>\s*)<\?xml\b[^?]*\?>", r"\1", xml_text)
        repaired = re.sub(r"^<\?xml\b[^?]*\?>\s*", "", repaired)
        try:
            wrapper = ET.fromstring(f"<wechat_message>{repaired}</wechat_message>")
        except ET.ParseError:
            return None
        for child in wrapper:
            if child.tag in {"msg", "sysmsg", "voipmsg", "recordinfo", "appmsg"}:
                return child
        return None


def first_text(root: ET.Element, *paths: str) -> str:
    for path in paths:
        value = find_text(root, path)
        if value:
            return value
    return ""


def parse_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_text(root: ET.Element | None, tag: str) -> str:
    if root is None:
        return ""
    value = root.findtext(tag) or ""
    return value.strip()


def compact_long_text(text: str, fallback: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return fallback
    return cleaned[:600] + ("..." if len(cleaned) > 600 else "")


def safe_media_path(media_root: Path, rel: str) -> Path | None:
    if not rel or rel.startswith(("/", "\\")):
        return None
    if any(part in {"", ".", ".."} for part in Path(rel).parts):
        return None
    try:
        root = media_root.resolve()
        path = (root / rel).resolve()
    except OSError:
        return None
    root_text = str(root)
    path_text = str(path)
    if path_text != root_text and not path_text.startswith(root_text + os.sep):
        return None
    return path


def decrypt_v2_image_data(data: bytes, aes_key: bytes | None, xor_key: int) -> bytes | None:
    if not data.startswith(V2_IMAGE_MAGIC):
        return data
    if not aes_key or AES is None or len(data) < 15:
        return None
    aes_size = int.from_bytes(data[6:10], "little")
    xor_size = int.from_bytes(data[10:14], "little")
    if aes_size > 100 * 1024 * 1024 or xor_size > 100 * 1024 * 1024:
        return None
    aes_ct_size = aes_size + 16 if aes_size % 16 == 0 else ((aes_size + 15) // 16) * 16
    aes_start = 15
    aes_end = aes_start + aes_ct_size
    raw_end = len(data) - xor_size if xor_size else len(data)
    if aes_end > raw_end or raw_end > len(data):
        return None
    try:
        plaintext = AES.new(aes_key, AES.MODE_ECB).decrypt(data[aes_start:aes_end])
    except Exception:
        return None
    out = bytearray(plaintext[:aes_size])
    out.extend(data[aes_end:raw_end])
    if xor_size:
        out.extend(byte ^ xor_key for byte in data[raw_end:])
    decoded = bytes(out)
    return decoded if sniff_mime(decoded, "") != "application/octet-stream" else None


def sniff_mime(data: bytes, name: str) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    return mimetypes.guess_type(name)[0] or "application/octet-stream"
