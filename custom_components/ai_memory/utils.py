from datetime import datetime


def format_date(iso_string):
    if not iso_string:
        return ""
    try:
        dt = datetime.fromisoformat(iso_string)
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso_string
