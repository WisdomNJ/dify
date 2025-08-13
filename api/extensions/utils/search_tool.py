import difflib
from collections import defaultdict, Counter
import itertools
import re
import jieba
import jieba.analyse
import json
from typing import Any, Optional, cast
import math

class Keywords:
    def __init__(self, texts, main_texts, search_texts, search_sql):
        self.texts = texts
        self.main_texts = main_texts
        self.search_texts=search_texts
        self.search_sql=search_sql

    def to_dict(self):
        return {
            "texts": self.texts,
            "main_texts": self.main_texts,
            "search_texts": self.search_texts,
            "search_sql": self.search_sql,
        }
class TextIndex:
    def __init__(self, text_text, index):
        self.text_text = text_text
        self.index = index

    def to_dict(self):
        return {
            "text_text": self.text_text,
            "index": self.index,
        }

def find_all_occurrences(source: str, target: str):
    return [match.start() for match in re.finditer(re.escape(source), target)]

def get_text_max_score(search_texts: list[str],search_index: int, pos_map,root_list:list[TextIndex], groups:list[list[TextIndex]]):

    if len(search_texts) == search_index and len(root_list) > 0:
        groups.append(root_list)
        return
    search_text = search_texts[search_index]
    text_indexs = pos_map[search_text]
    next_index = search_index + 1
    if text_indexs:
        new_root_list = root_list[:]
        for t_idx,text_index in enumerate(text_indexs):
            this_root_list = []
            if t_idx > 0:
                this_root_list=new_root_list[:]
            else:
                this_root_list = root_list
            this_root_list.append(text_index)
            get_text_max_score(search_texts=search_texts,search_index=next_index,pos_map=pos_map,root_list=this_root_list,groups=groups)
    else:
        get_text_max_score(search_texts=search_texts,search_index=next_index,pos_map=pos_map,root_list=root_list,groups=groups)

# def get_text_index_score(text_indexs: list[TextIndex],search_texts: list[str]):
#     # 去掉一个最后面的
#     # 去掉一个最前面的

def get_text_index_score(text_indexs: list[TextIndex],search_texts: list[str]):

    deduct_points = 0
    text_count = 0
    for idx,text_index in enumerate(text_indexs):
        text_count += len(text_index.text_text)
        if idx < len(text_indexs) - 1:
            next_text_index = text_indexs[idx + 1]
            t_score = 0
            if next_text_index.index > text_index.index:
                t_score = next_text_index.index - text_index.index - len(text_index.text_text) - 1
            else:
                t_score = text_index.index - next_text_index.index - len(next_text_index.text_text)
            t_score = abs(t_score)
            deduct_points += t_score
            if deduct_points > 50:
                return 0
    search_text_count = len("".join(search_texts))
    deduct_points += (search_text_count - text_count) * 3
    return 100 - deduct_points

def get_full_search_text_max_score(search_texts: list[str], target_text: str) -> (int, list[TextIndex]):
    # 1. 建立 source 中每个字符的索引映射
    # pos_map = defaultdict(list)
    text_index_groups:list[list[TextIndex]] = []
    for search_text in search_texts:
        idxs = find_all_occurrences(source=search_text, target=target_text)
        text_indexs = [TextIndex(text_text=search_text,index=idx) for idx in idxs]
        # pos_map[search_text].extend(text_indexs)
        text_index_groups.append(text_indexs)

    # groups:list[list[TextIndex]] = []
    max_score = -100000
    max_index_list:list[TextIndex]=[]
    for text_index_s in itertools.product(*text_index_groups):
        text_index_list:list[TextIndex] = list(text_index_s)
        score_ = get_text_index_score(text_indexs=text_index_list,search_texts=search_texts)
        if score_ < 80:
            continue
        if score_ > max_score:
            max_score = score_
            max_index_list = text_index_list

    # get_text_max_score(search_texts=search_texts,search_index=0,pos_map=pos_map,root_list=[], groups=groups)
    # max_index_list:list[TextIndex] = []
    # max_score = -100000
    # import pdb; pdb.set_trace()
    # for g_list in groups:
    #     score_,milist = get_text_index_score(text_indexs=g_list,search_texts=search_texts)
    #     if score_ > max_score:
    #         max_score = score_
    #         max_index_list = g_list
    #     print("score_",score_)
    #     texts = []
    #     for text_index in g_list:
    #         t_len = len(text_index.text_text)
    #         t_idx = text_index.index
    #         text = target_text[t_idx : t_idx+t_len]
    #         texts.append(text)
    #     print("--------------------------")
    #     print("".join(texts))
    return (max_score,max_index_list)

def get_main_keywords_texts(query_text: str) -> list[str]:
    # 判断关键词的长度
    jieba.analyse.set_stop_words("extensions/utils/stopwords.txt")
    # jieba.analyse.set_idf_path("extensions/utils/idfwords.txt")
    # 提取关键词，默认 topK=30，withWeight=True
    main_keywords_texts__ = jieba.analyse.extract_tags(query_text, topK=200, withWeight=False)

    return main_keywords_texts__

def get_keywords(query_text: str) -> Keywords:
    # 分词器分词关键词
    keyword_texts = list(jieba.cut(query_text))
    keyword_texts_for_search = list(jieba.cut_for_search(query_text))
    print("keyword_texts:",keyword_texts)
    print("keyword_texts_for_search:",keyword_texts_for_search)
    main_keywords_texts__ = get_main_keywords_texts(query_text=query_text)
    print("main_keywords_texts__:",main_keywords_texts__)
    keyword_len = len(main_keywords_texts__)
    main_keywords_len = 0
    # import pdb; pdb.set_trace()
    # 提取80%
    if keyword_len > 4:
        main_keywords_len = int(keyword_len * 0.8)
    else:
        main_keywords_len = keyword_len

    main_keywords_len = keyword_len if main_keywords_len > keyword_len else main_keywords_len
    # 得出最关键的分词
    search_keywords_texts__ = main_keywords_texts__[:main_keywords_len]

    main_keywords_texts = []
    search_keywords_texts = []
    for text in keyword_texts:
        if text in main_keywords_texts__:
            main_keywords_texts.append(text)
        if text in search_keywords_texts__:
            search_keywords_texts.append(text)

    search_sql = get_search_keywords_texts_sql(search_keywords_texts=search_keywords_texts)
    # search_sql = ' & '.join(search_keywords_texts)
    # 按照最关键的分词查询
    keywords = Keywords(
        texts=main_keywords_texts,
        main_texts=main_keywords_texts,
        search_texts=search_keywords_texts,
        search_sql=search_sql
    )
    return keywords

def get_search_keywords_texts_sql(search_keywords_texts:list[str]):

    texts = []
    query_sql_list = []
    for text in search_keywords_texts:
        # 将元素才拆成可查询用的分词
        texts_for_search:list[str] = list(jieba.cut_for_search(text))

        search_texts = [*texts_for_search]
        if text not in search_texts:
            search_texts.append(search_texts)
        query_sql_list.append(f"({" | ".join(search_texts)})")

        min_texts:list[str] = get_min_search_keywords_texts(texts=texts_for_search)
        texts.extend(min_texts)
    query_sql = " & ".join(query_sql_list)
    # import pdb; pdb.set_trace()
    texts_len = len(texts)
    sql = ""
    if texts_len == 1:
        sql = texts[0]
    elif texts_len == 2:
        merge_text = merge_strings(texts[0],texts[1])
        sql = f"{texts[0]} & {texts[1]} | {merge_text}"
    else:
        sql_texts:list[str] = []
        for idx,text in enumerate(texts):
            if idx == 0:
                merge_text = merge_strings(text,texts[idx + 1])
                sql_texts.append(f"({text} | {merge_text})")
            elif idx == texts_len - 2:
                merge_text1 = merge_strings(text,texts[idx + 1])
                merge_text2 = merge_strings(texts[idx-1],text)
                sql_texts.append(f"({text} | {merge_text1} | {merge_text2} & {texts[idx + 1]})")
            elif idx == texts_len - 1:
                merge_text = merge_strings(texts[idx-1],text)
                sql_texts.append(f"({text} | {merge_text})")
            else:
                merge_text1 = merge_strings(text,texts[idx + 1])
                merge_text2 = merge_strings(texts[idx-1],text)
                merge_text3 = merge_strings(texts[idx + 1],texts[idx + 2])
                sql_texts.append(f"({text} | {merge_text1} | {merge_text2} & ({texts[idx + 1]} | {merge_text3}))")
        sql = " & ".join(sql_texts)
    print(sql)

    return f"({sql}) | ({query_sql})"

def merge_strings(text1, text2):
    max_overlap = 0
    min_len = min(len(text1), len(text2))

    # 找出最大重叠部分
    for i in range(1, min_len + 1):
        if text1[-i:] == text2[:i]:
            max_overlap = i
    # 合并字符串
    text = text1 + text2[max_overlap:]
    return text

def get_min_search_keywords_texts(texts:list[str]):
    # import pdb; pdb.set_trace()
    min_texts = []
    for text in texts:
        b = True
        for text2 in texts:
            if text != text2 and text2 in text:
                b = False
        if b:
            min_texts.append(text)
    return min_texts

# 扩展处理分值（全文检索的方法需要处理分值）
def set_full_search_score(query:str,doc_list:list[dict[str, Any]]):
    # 根据查询条件的长短
    main_keywords_texts = get_main_keywords_texts(query_text=query)

    all_texts = []
    for main_keywords_text in main_keywords_texts:
        keyword_texts_for_search = list(jieba.cut_for_search(main_keywords_text))
        all_texts.extend(keyword_texts_for_search)

    sum_lens = len(all_texts)
    sum_lens = 2 if sum_lens == 1 else sum_lens
    plus_score = score(sum_lens)
    print("plus_score",plus_score)
    if doc_list:
        for doc in doc_list:
            metadata = doc["metadata"]
            if metadata:
                dataset_name = metadata["dataset_name"]
                doc_score = metadata["score"]
                if dataset_name == "FULL_TEXT_SEARCH_KNOWLEDGE" and doc_score:
                    doc_score += plus_score
                    doc["metadata"]["score"] = doc_score
                    print("new score:",doc["metadata"]["score"])
        for doc in doc_list:
            if doc["metadata"] and doc["metadata"]["score"]:
                print("new score:",doc["metadata"]["score"])

def score(value):
    return round(20 * math.exp(-0.4 * value), 2) / 100

def get_main_keywords_texts_test(query_text: str) -> list[str]:
    # 判断关键词的长度
    jieba.analyse.set_stop_words("d://stopwords.txt")
    jieba.analyse.set_idf_path("extensions/utils/idfwords.txt")
    # 提取关键词，默认 topK=30，withWeight=True
    main_keywords_texts__ = jieba.analyse.extract_tags(query_text, topK=200, withWeight=False)

    return main_keywords_texts__

# 自定义的同义词词典（示例）
default_synonym_dict = {
    "回款": ["收款","收款金额", "款项", "支付", "款项回收", "回款金额","回款情况"],
    "查看": ["查询","查找"],
    "类型": ["类型","型号","种类","类别","分类"],
    "级别": ["级别","等级"],
    "形式": ["形式","模式","样式"],
    "执行": ["关闭","售前", "激活"],
    "名称": ["代号","标题", "命名","称号"],
    "哪些": ["哪些","多少","有哪些","有多少","列举","清单","列表","记录","明细"],
    "耗费" : ["消耗","耗费","花费","消费掉"],
    "工时": ["消耗工时","项目工时","耗费工时","耗费工时","花费工时","消费掉工时","投入工时"],
    "项目": ["激活项目", "关闭项目", "售前项目", "执行项目"],
    "指定日期": ["本周", "本月", "上个月", "上周", "上个星期",
                 "上月", "上个月", "下月", "前一周", "过去一周", "最近一周",
                 "七天前", "未来一个月", "下一月", "一个月后", "近三月","这个月",
                 "今年","去年","上半年","下半年","明年",
                 "上一年","2025年","2024年"],
    "异常": ["异常","不匹配", "滞后","延期","推迟","延迟","滞缓","拖延"],
    "各项目": ["各项目","每个项目","各个项目"],
    "计划模板": ["计划模板","任务模板","项目模板","作业模板"],
}

biz_synonym_dict = {
    "某某类型-计划模板": ["条件建设","民用","军机","科研"],
    "某某类型-WBS模板": ["条件建设","民用","军机","科研"],
    "某某计划模板": ["攻关计划模板","科技研究项目"],
    "某某WBS模板": ["攻关计划模板","科技研究项目"],
    "某某项目": ["某某项目","某地桥梁道路建设","A发动机研制项目","新版ACM移动端开发","航空工业集团XX项目管理系统导航项目","132厂综合计划管理系统","118厂多项目管理系统","航发成发多项目管理系统","航发商发跨厂所供应商协同研制管理研究","航发集团重点计划管理系统","航发商发新产品导入过程（NPI）方法工具研究","航发商发技术成熟度综合评价系统技术研究","航发商发协同平台项目","航发商发企业资源计划项目（经营计划管理）","航发商发企业资源计划项目（项目管理和合同管理）","航发商发质量数据的分析与归纳项目","132厂项目管理系统开发","601所多管理体系融合项目","132厂固定资产投资管理系统","601所项目管理系统工作量控制功能优化项目","132厂任务管理系统","719所质量检验与计量管理平台","719所门户基础应用及用户管理","719所船舶控制站运动仿真及视景仿真软件开发","704所制造项目计划协同子系统项目","719所项目管理系统优化升级","中核华辉进度计划管理系统","福清核电大修计划自动排程软件研究开发","苏轨院工程项目信息化管理系统","132厂多维度综合项目管理系统","测试费用","维斯德渠道发展计划","测试一下","办公楼施工计划案例-bak","测试主管项目","611所项目管理系统","602所项目管理系统","办公楼施工计划案例","AVIC总部(KD)敏捷协同管理系统","132厂计划预算一体化管控平台","ACM小程序试点项目","ACM移动端试点","维斯德敏捷协同管理平台软件V3.0","维斯德资源管理系统软件V2.0","维斯德文档管理系统软件V2.0","维斯德风险管理平台软件V2.0","智能建造全寿期数据处理分析管控软件V2.0","平台组知识图谱","平台组Wisdom+开发平台","稳岗补贴2021","软件企业年度评估申报","维斯德雨花台区软件产业专项资金申请（资质认证奖励1）","维斯德雨花台区软件产业专项资金申请（产学研）","维斯德雨花台区软件产业专项资金申请（研发投诉奖励）","维斯德雨花台区软件产业专项资金申请（房屋租赁补贴1）","维斯德雨花台区软件产业专项资金申请（软件人才综合补贴）","维斯德资质申报2021","维斯德课题申报2021","敏捷协同管理平台（ACM）- 2022","敏捷协同管理平台（ACM）移动端","132厂党建计划管理系统","知识图谱项目文档权限控制","AVIC总部企业级敏捷项目管理系统(一期)","测试一下-01","商发ERP平台运维服务（2022年度）","费控修改","云南华联锌铟工程项目管理系统","132厂项目管理系统运维","中国卫星网络公司项目","上海集塔项目","智能制造全寿期数据管理平台的研发-全文检索","维斯德ACM敏捷协同管理平台-风险管理","2月18日至3月17日项目攻坚","2022年项目例会行动项","福清核电采购及合同管理系统","连云港田湾核电检修售前","天码智控项目售前","中核—售前","福清核电预防性项目数据库辅助优化服务合同","132厂固定资产二期","弱电工程","一号楼弱电工程","演示项目1","测试项目","电站建设项目","河北XX风电项目","计划demo","演示计划","北京XX风电项目","XXX-A区200MW风电项目","XXX-B区200MW风电项目","龙田项目","Cairo","股份公司2021年度固定资产投资江苏项目","敏捷协同管理ACM移动端推广","大学生家居用品店","猫千岁推广活动","自媒体创业经历","中国电子集团电子六所业务管理系统","测试项目","测试项目1","演示项目1","招聘项目经理","大丰酒店项目","卓越浦口","正荣紫阙","王路安项目","研发项目","大理石门楼建设","北固湾","MOM装配","MOM零件","MOM资源","MOM库存","MOM检验","电缆执行","TR2 ","T3高杆灯改造","法兰谷 178-2-303","萨家湾39号金川雅居1-3-1505","绿地风情雅苑8-1602","大理幸福城华庭北园16-1-601","万科光明城市14-602","敏捷协同管理系统","星链一号","DEMO测试项目","X改扩建项目","2022XX站大修项目","A型矿山智能机械控制软件研发","安质专项建设项目","某部门电机研发立项项目","中核计算机所（北京）项目售前","北京规划院","长城汽车","北京兵器装备集团","四川长虹","航空工业集团合同管理","601所多项目系统管理系统优化提升项目","天桥嘉成项目","南方航宇高精传动项目","公司注册以及银行税务开户","信创环境备份与恢复项目","西飞密标项目","南阳某兵器公司mes项目","2022年公司管理专项","公司开业仪式会议","公司软件著作权申请","图形工作站","西飞密标项目2","质量文档管理系统","天光WMS项目","公司MES产品研发项目管理","182钣金总厂MOM项目","装备合同监管业务管控中心软件","3207数字化车间","公司APS产品研发项目","2022-2023年发展规划编制","2022年公司销售售前项目跟踪","德兰航宇MES项目","西安公司注册及办公室租赁","集中建设单位考评","生马741线路整修项目","组织活动","丽水大会展泛光照明项目","演示项目","天玛科技管理信息化系统","金航&维斯德移交项目","AR940制造BOM数据与供应商系统集成接口开发服务","项目管理模板示例","工业软件目录测评","敏捷集成平台（AIP）","北京航空材料研究院多项目管理系统开发","中国原子能科学研究院全面项目管理系统","成飞航产CPM系统采购项目","2023项目沟通","市场拓展项目—成都办","市场拓展项目—北京办","市场拓展项目—上海办","榆林有色集团公司数字化转型项目","贵飞CPM安全认证集成及三员认证功能开发","洛阳014中心空空导弹研究院","中交集团项目","602所项目管理系统二期","陪标项目","中国工程物理研究院售前项目","项目实施标准化示例项目","敏捷协同管理系统（AVIC）","敏捷协同管理系统（ACM）","敏捷开发平台（ADP）","2023年度ERP平台系统运维服务","118厂军机项目管理系统","综合部日常事务","132厂所属企业Wisdom平台软件采购","南京机电(609所)项目管理系统","607所项目管理系统一期","航材院科研生产协同管控中心开发","航材院AEOS服务保障","三一筑享云项目","中航无人机党建计划管理项目","基于华为云的Welink软件项目","合同管理功能改进项目","项目计划管理功能改进项目","三一集团iPMS项目","611所项目管理系统二期","南京机电科技项目管理系统","中核华辉工程进度计划管理系统技术开发项目","AVIC总部行业项目协同管理服务组件系统(二期)","601多项目协同管理系统项目管理要素功能补充开发项目","607所项目管理系统二期","2024年年度会议行动项","2024年度ERP平台系统运维服务","132厂多维度综合项目管理系统优化改造","132厂综合计划管理系统改造","601薪酬预算管理系统","市场拓展项目—南京本部","AVIC总部项目增补(巡视整改、专项工程、条件建设、督办项)","敏捷协同管理系统(ACM)V4.0","test测试","国营四达综合计划管理系统项目","611所项目管理系统项目分解项目(补充合同)","航材院项目增补","新乡航空项目管理系统","158洛阳光电项目管理系统","IPD集成产品开发","企业经营管理系统V1.0","成飞航产CPM系统运维项目","测试项目A","(9院8所)中国工程物理研究院激光聚变研究中心项目","5720厂项目管理系统项目","AI检索新增项目测试","AVIC总部项目定制化开发","JZ系统项目管理系统","三一海工云项目平台","5713综合计划管理系统项目","航发集团项目管理系统","商发产品合格认证系统与供应商系统集成开发项目","原子能全面项目管理系统运维服务","132厂规划管理系统项目","企业经营管理系统","企业战略管理系统","预研与大模型整合","市场拓展项目—事业一部","市场拓展项目—事业二部","市场拓展项目—市场营销部","市场拓展项目—南京本部","602所项目管理系统三期","132厂党建计划管理系统二期","任务流程管理与仿真数据管理开发服务项目","葫芦岛海军基地项目管理系统","原子能全面项目管理系统迁移部署","测试项目","XXX项目","航发503计算机采购","天奥电子mes ","西飞620厂智能仓储配送系统","榆林新材料集团数字化转型项目","项目管理","演示项目1","项目1","自研项目A","XX型号项目","歼-XXX飞机","内部测试","ffffff","测试项目","Test 项目协同","陕西有色榆林新材料集团有限责任公司《集团公司数字化转型项目建设与应用》一期项目建设总集服务项目","成都天奥电子股份有限公司制造执行系统（MES）项目一期","2023年领航智造管理专项","X公司系统集成硬件询价","2023年售前项目管理","测试","演示项目","123","1234","12345","369","测试项目团队","测试部门","项目管理模板生成","项目管理模板生成1","项目管理模板创建","演示项目3","IT 系统实施1","空白项目","项目管理演示","某军工项目","KLIA APM project","诗天下","诗天下啊","安吉成本管理","葫芦娃没有爷爷","维斯德","ZJ综合计划考核项目","崇实里配套道路","JZ项目咨询","项目1","A型号项目-测试","B型号项目-测试","C型号项目-测试","测试项目0815","测试立项"],
}

new_words = ["某某计划模板","激活项目", "关闭项目", "售前项目", "执行项目","航空工业项目","602所","指定日期","某某项目","回款计划","项目工时","回款情况","消耗工时","不匹配","有哪些"]
# 1. 中文分词
def segment_text(text):
    main_keywords_texts__ = jieba.analyse.extract_tags(text, topK=200, withWeight=False)
    return list(main_keywords_texts__)
    # return list(jieba.cut(text))

def add_words(words:list[str]):
    if words:
        for word in words:
            # print("----", word)
            jieba.add_word(word,freq=2000, tag='n')

def add_words_ner(words:list[str],init_freq:int):
    if words:
        for word in words:
            # print("----", word)
            freq = init_freq + len(word)
            jieba.add_word(word,freq=freq, tag='n')

# 2. 同义词替换函数
def replace_synonyms(text,synonym_dict):
    # 分词
    words = segment_text(text)
    for i, word in enumerate(words):
        for key, synonyms in synonym_dict.items():
            if word in synonyms:
                words[i] = key  # 用标准词替换
                break
    return words

def contains_placeholder_date(s):
    patterns = [
        r'\d{4}年\d{1,2}月\d{1,2}日',
        r'\d{4}-\d{1,2}-\d{1,2}',
        r'\d{4}/\d{1,2}/\d{1,2}',
    ]
    matches_d = []
    for pattern in patterns:
        a =  bool(re.search(pattern, s))
        if a:
            # 使用 findall 提取所有匹配
            matches_d = re.findall(pattern, s)

    patterns = [
        r'\d{4}年\d{1,2}月',
        r'\d{4}-\d{1,2}',
        r'\d{4}/\d{1,2}',
    ]
    matches_m = []
    for pattern in patterns:
        a =  bool(re.search(pattern, s))
        if a:
            # 使用 findall 提取所有匹配
            matches_m = re.findall(pattern, s)
    patterns = [
        r'\d{4}年'
    ]
    matches_y = []
    for pattern in patterns:
        a =  bool(re.search(pattern, s))
        if a:
            # 使用 findall 提取所有匹配
            matches_y = re.findall(pattern, s)

    return "指定日期" , matches_d + matches_m + matches_y

def contains_placeholder_month(s):
    patterns = [
        r'\d{4}年\d{1,2}月',
        r'\d{4}-\d{1,2}',
        r'\d{4}/\d{1,2}',
    ]
    for pattern in patterns:
        a =  bool(re.search(pattern, s))
        if a:
            # 使用 findall 提取所有匹配
            matches = re.findall(pattern, s)
            return True,matches
    return False,[]

def contains_placeholder_year(s):
    patterns = [
        r'\d{4}年\d{1,2}月',
        r'\d{4}-\d{1,2}',
        r'\d{4}/\d{1,2}',
    ]
    for pattern in patterns:
        a =  bool(re.search(pattern, s))
        if a:
            # 使用 findall 提取所有匹配
            matches = re.findall(pattern, s)
            return True,matches
    return False,[]

def contains_placeholder_date2(s):
    patterns = [
        r'前\d{1}周',
        r'\d{4}-\d{1,2}-\d{1,2}',
        r'\d{4}/\d{1,2}/\d{1,2}',
    ]

    for pattern in patterns:
        a =  bool(re.search(pattern, s))
        if a:
            # 使用 findall 提取所有匹配
            matches = re.findall(pattern, s)
            return True,matches
    return False,[]

def handle_question(question:str,word_list:list,synonym_dict:dict):
    flg,matches = contains_placeholder_date(question)
    if flg:
        key = "指定日期"
        if key in synonym_dict:
            ds = synonym_dict[key]
            matches = ds + matches
        synonym_dict[key] = matches
        word_list.extend(matches)

def add_words_ner_by_key_words(key_words:list[str],init_freq:int,synonym_dict:dict):
    biz_synonyms = []
    for word in key_words:
        if word in biz_synonym_dict:
            synonyms = biz_synonym_dict[word]
            word_arr = word.split("-")
            word_text = word_arr[0]
            if synonyms and len(synonyms) > 0:
                biz_synonyms.extend(synonyms)
                biz_synonyms.append(word_text)
                values = synonyms
                if word in synonym_dict:
                    ds = synonym_dict[word]
                    if ds:
                        values = ds + values
                synonym_dict[word] = values
    add_words_ner(biz_synonyms,init_freq)

# 3. 设置名词
def set_words(target_required:str,
              target_un_required: str,
              word : str,
              synonym_dict:dict):

    word_list = word.split(",") if word else []
    # 自定义名词
    add_words_ner(word_list,2500)
    # 通用名词
    set_word_by_synonym_dict(synonym_dict=synonym_dict)

    required_words = target_required.split(",")
    # 必填业务数据名词
    add_words_ner_by_key_words(required_words,2000,synonym_dict)
    # 非必填的业务数据名词
    un_required_words = target_un_required.split(",")
    add_words_ner_by_key_words(un_required_words,2000,synonym_dict)

def set_word_by_synonym_dict(synonym_dict:dict):
    word_list = []
    if synonym_dict:
        for key in synonym_dict.keys():
            ds = synonym_dict[key]
            if ds:
                word_list.append(key)
                word_list.extend(ds)
    add_words(word_list)

def set_synonym_by_question(question:str,synonym_dict:dict):
    # 指定日期
    key,matches = contains_placeholder_date(question)
    set_synonym_dict(key,matches,synonym_dict)


def set_synonym_dict(key:str,matches:list[str],synonym_dict:dict):
    if len(synonym_dict) > 0:
        if key in synonym_dict:
            ds = synonym_dict[key]
            matches = ds + matches
        synonym_dict[key] = matches

# 获取同义词，合并自定义的同义词
def get_synonym_dict(synonym:str):

    new_synonym_dict:dict = {
        **default_synonym_dict
    }
    if synonym:
        synonym_dict = json.loads(synonym)
        for key in synonym_dict.keys():
            original = synonym_dict[key]
            values = original.split(",") if original else []
            if key in new_synonym_dict:
                ds = new_synonym_dict[key]
                if ds:
                    values = ds + values
            new_synonym_dict[key] = values
    return new_synonym_dict

# 3. 计算模糊匹配分数
def api_desc_match(question_text:str,
                   name : str,
                   target_required:str,
                   target_un_required: str,
                   word : str,
                   synonym : str,):
    jieba.analyse.set_stop_words("extensions/utils/stopwords.txt")
    # jieba.analyse.set_stop_words("d://stopwords.txt")
    # 获取同义词
    synonym_dict = get_synonym_dict(synonym)
    # 根据问题，追加同义词
    set_synonym_by_question(question=question_text,synonym_dict=synonym_dict)
    # 设置名词
    set_words(target_required=target_required,target_un_required=target_un_required,word=word,synonym_dict=synonym_dict)
    # 判断是否匹配
    return is_match(question_text=question_text,name=name,target_required=target_required,target_un_required=target_un_required,synonym_dict=synonym_dict)

def is_match(question_text:str,
             name : str,
             target_required:str,
             target_un_required: str,
             synonym_dict: dict):

    # 替换同义词
    query_list = replace_synonyms(question_text,synonym_dict)
    query_list = [ a.split("-")[0] for a in query_list]

    target_required_list = replace_synonyms(target_required,synonym_dict)
    target_un_requiredlist = replace_synonyms(target_un_required,synonym_dict)
    print("---------------",name)
    print("query_list",query_list)
    print("target_required_list",target_required_list)
    print("target_un_requiredlist",target_un_requiredlist)
    # 判断必要关键字是否全部匹配
    if is_all_contains_key_words(query_list=query_list,target_list=target_required_list):
        # 去除必要关键字
        last_list = list(set(query_list) - set(target_required_list))
        # 判断c是否还有数据
        if len(last_list) > 0:
            # 从条件内去除选填的key
            last_list = list(set(last_list) - set(target_un_requiredlist))
        if len(last_list) == 0:
            return True
    return False

# 判断是否全部关键分词
def is_all_contains_key_words(query_list : list[str], target_list : list[str] ) -> bool:

    # 先按长度从长到短排序，长的更可能是“父串”
    sorted_target_list = sorted(target_list, key=len, reverse=True)
    result = []
    id_contains = True
    for target_key in sorted_target_list:
        a = target_key in query_list
        b = any(target_key in r for r in result)
        if a or b:
            result.append(target_key)
        else:
            id_contains = False
            break
    return id_contains

    # 过滤包含的key
    # list_str = filter_contained_strings(list_str=target_list)
    # if len(list_str) > 0:
    #     if set(list_str).issubset(set(query_list)):
    #         return True
    # return False

def filter_contained_strings(list_str:list[str]):
    result = []
    if list_str:
        # 先按长度从长到短排序，长的更可能是“父串”
        sorted_list_str = sorted(list_str, key=len, reverse=True)
        for s in sorted_list_str:
            # 检查当前字符串是否被 result 中已保留的任何一个字符串包含
            if not any(s in other and s != other for other in sorted_list_str):
                result.append(s)
    print("filter_contained_strings",result)
    return result

# 判断是否匹配
# def api_desc_match_old(question_text:str,
#                    name : str,
#                    target_required:str,
#                    target_un_required: str,
#                    word : str,
#                    synonym : str,):
#
#     # jieba.analyse.set_stop_words("extensions/utils/stopwords.txt")
#     jieba.analyse.set_stop_words("d://stopwords.txt")
#
#     word_list = word.split(",") if word else []
#     new_synonym_dict:dict = {**default_synonym_dict}
#     if synonym:
#         synonym_dict = json.loads(synonym)
#         for key in synonym_dict.keys():
#             original = synonym_dict[key]
#             values = original.split(",") if original else []
#             if key in new_synonym_dict:
#                 ds = new_synonym_dict[key]
#                 if ds:
#                     values = ds + values
#             new_synonym_dict[key] = values
#
#     word_list = new_words + word_list
#
#     handle_question(question=question_text,word_list=word_list,synonym_dict=new_synonym_dict)
#
#     for key in new_synonym_dict.keys():
#         ds = new_synonym_dict[key]
#         if ds:
#             word_list.extend(ds)
#
#     add_words(word_list)
#     # 先替换同义词
#     query_list = replace_synonyms(question_text,new_synonym_dict)
#     target_required_list = replace_synonyms(target_required,new_synonym_dict)
#     target_un_requiredlist = replace_synonyms(target_un_required,new_synonym_dict)
#
#     print("---------------",name)
#     print("query_list",query_list)
#     print("target_required_list",target_required_list)
#     print("target_un_requiredlist",target_un_requiredlist)
#     # 判断必要关键字是否全部匹配
#     if set(target_required_list).issubset(set(query_list)):
#         # 去除必要关键字
#         last_list = list(set(query_list) - set(target_required_list))
#         # 判断c是否还有数据
#         if len(last_list) > 0:
#             # 从条件内去除选填的key
#             last_list = list(set(last_list) - set(target_un_requiredlist))
#         if len(last_list) == 0:
#             return True
#     return False

if __name__ == "__main__":
    # 示例
    query = "查看科研型号的计划模板"
    target = "指定日期的回款计划"
    target_list = [
        "指定日期的回款计划",
        "指定日期回款金额",
        "回款与消耗工时不匹配的项目"
    ]
    synonym = "{\"指定日期\" : \"FFFF,GGGG,EEE\" }"
    # 获取模糊匹配分数
    match_score = api_desc_match(query,
                                 name="查看名称叫：攻关计划模板的模板",
                                 target_required="某某类型-计划模板,某某类型,类型,计划模板",
                                 target_un_required="计划,查询,哪些,什么,项目,查看,清单,名称",
                                 word="",
                                 synonym=synonym )
    print(f"模糊匹配分数：{match_score}")

    # print(merge_strings("第二","二层"))
    # get_keywords("我的")
    # search_texts=["湖人","阵容"]
    # score, max_index_list =get_full_search_text_max_score(search_texts=search_texts, source="所以，**严格讲，詹姆斯在湖人确实拥有超级巨星（戴维斯），但不像热火三巨头那样多核并立。**更多时候，他还是湖人阵容的绝对核心和领袖。")
    # print(score, len(max_index_list))
