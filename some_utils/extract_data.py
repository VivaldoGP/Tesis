import datetime


def date_from_filename(filename: str, final_part: int = -3):
    try:
        date_part = filename.split('_')[final_part]
        year: int = int(date_part[:4])
        month: int = int(date_part[4:6])
        day: int = int(date_part[6:8])
        return datetime.date(year=year, month=month, day=day)
    except (TypeError, ValueError):
        return None
