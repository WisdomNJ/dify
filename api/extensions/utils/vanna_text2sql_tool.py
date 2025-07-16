import re


def handle_sql(sql : str):
    if sql:
        # 去除单行注释（以 -- 开头的注释）
        cleaned_sql = re.sub(r'--.*', '', sql)
        # 去除多余空行
        cleaned_sql = '\n'.join(line for line in cleaned_sql.splitlines() if line.strip())
        return cleaned_sql
    return sql

if __name__ == "__main__":
    sql = """
WITH
-- 生成1-12月的月份序列
months AS (
    SELECT generate_series(1, 12) AS month_num
),
-- 计算每月计划回款金额
plan_data AS (
    SELECT EXTRACT(MONTH FROM f.plan_end_time) AS month_num,
           SUM(f.plan_return_money) AS plan_amount
    FROM wsd_plan_project_return_fund f
    JOIN wsd_plan_project p ON f.project_id = p.id
    WHERE EXTRACT(YEAR FROM f.plan_end_time) = EXTRACT(YEAR FROM CURRENT_DATE)
      AND p.del = 0 AND f.del = 0 AND p.status != '关闭'
    GROUP BY month_num
),
-- 计算每月实际回款金额
actual_data AS (
    SELECT EXTRACT(MONTH FROM f.act_end_time) AS month_num,
           SUM(f.returned_money) AS actual_amount
    FROM wsd_plan_project_return_fund f
    JOIN wsd_plan_project p ON f.project_id = p.id
    WHERE EXTRACT(YEAR FROM f.act_end_time) = EXTRACT(YEAR FROM CURRENT_DATE)
      AND p.del = 0 AND f.del = 0 AND p.status != '关闭'
    GROUP BY month_num
)
-- 最终结果展示
SELECT m.month_num || '月' AS "月份",
       COALESCE(p.plan_amount, 0) AS "计划回款金额(元)",
       COALESCE(a.actual_amount, 0) AS "实际回款金额(元)",
       CASE
           WHEN COALESCE(p.plan_amount, 0) = 0 THEN '0%'
           ELSE ROUND(COALESCE(a.actual_amount, 0) / COALESCE(p.plan_amount, 1) * 100, 2) || '%'
       END AS "完成率",
       COALESCE(p.plan_amount, 0) - COALESCE(a.actual_amount, 0) AS "差额(元)"
FROM months m
LEFT JOIN plan_data p ON m.month_num = p.month_num
LEFT JOIN actual_data a ON m.month_num = a.month_num
ORDER BY m.month_num;
"""
    # print(merge_strings("第二","二层"))
    new_sql = handle_sql(sql)
    print(new_sql)
    # search_texts=["湖人","阵容"]
    # score, max_index_list =get_full_search_text_max_score(search_texts=search_texts, source="所以，**严格讲，詹姆斯在湖人确实拥有超级巨星（戴维斯），但不像热火三巨头那样多核并立。**更多时候，他还是湖人阵容的绝对核心和领袖。")
    # print(score, len(max_index_list))
