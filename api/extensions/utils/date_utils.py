from datetime import datetime, timedelta, date

def get_today():
    """获取今天的日期"""
    today = date.today()
    return today.strftime('%Y-%m-%d')
def get_yesterday():
    """获取昨天的日期"""
    today = date.today()
    yesterday = today - timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')
def get_this_week_start():
    """获取本周的开始日期（周一）"""
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.strftime('%Y-%m-%d')
def get_this_week_end():
    """获取本周的结束日期（周日）"""
    today = date.today()
    sunday = today + timedelta(days=6 - today.weekday())
    return sunday.strftime('%Y-%m-%d')
def this_year_start():
    """获取今年的开始日期"""
    today = date.today()
    start_date = today.replace(month=1, day=1)
    return start_date.strftime('%Y-%m-%d')
def this_year_end():
    """获取今年的结束日期"""
    today = date.today()
    end_date = today.replace(month=12, day=31)
    return end_date.strftime('%Y-%m-%d')
def this_month_start():
    """获取本月的开始日期"""
    today = date.today()
    start_date = today.replace(day=1)
    return start_date.strftime('%Y-%m-%d')
def this_month_end():
    """获取本月的结束日期"""
    today = date.today()
    next_month = today.replace(day=28) + timedelta(days=4)
    return next_month.strftime('%Y-%m-%d')
def this_last_year(year: int = 1):
    """获取去年的开始日期"""
    today = date.today()
    last_year = today.replace(year=today.year - year)
    return last_year.strftime('%Y')


