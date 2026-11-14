"""Negative filters use verified identities, never keyword-based routing."""


def add_exclusion_filters(filters, params, scope, prefix=''):
    ids = (scope or {}).get('exclude_person_ids') or []
    if ids:
        marks = ','.join('?' for _ in ids)
        # A private conversation belongs to its peer, including our own replies.
        # In groups only that person's messages are excluded, not the whole group.
        filters.append(f"({prefix}chat_type != 'private' OR COALESCE({prefix}person_id,'') NOT IN ({marks}))")
        params.extend(ids)
        filters.append(f"COALESCE({prefix}sender_username,'') NOT IN ({marks})")
        params.extend(ids)
