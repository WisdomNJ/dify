import os
import json
from vanna.ollama import Ollama
from vanna.qianwen import QianWenAI_Chat
from vanna.deepseek import DeepSeekChat

from configs import dify_config
from extensions.utils.rewrite_ask import ask
from dotenv import load_dotenv
from vanna.milvus import Milvus_VectorStore
from pymilvus import DataType, MilvusClient,model
from collections import defaultdict
from extensions.utils.userclient import UserClient
from types import SimpleNamespace
from openai import OpenAI
import uuid

from models import Embedding

load_dotenv()
# 设置显示后端为浏览器
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List
import ollama
import numpy as np
from pymilvus.model.base import BaseEmbeddingFunction
# 自定义嵌入式模型（适配milvus向量数据库）
class CustomEmbeddingFunction(BaseEmbeddingFunction):
    def __init__(self):
        self.embed_model = dify_config.VANNA_EMBEDDING_MODEL
        self.embedding_model = OpenAI(
            base_url=dify_config.VANNA_EMBEDDING_HOST,
            api_key=dify_config.VANNA_EMBEDDING_API_KEY,
        )

    def __call__(self, texts: List[str]):
        self._encode(texts)

    def _encode(self, texts: list[str]) -> list[Embedding]:
        embeddings = self.embedding_model.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [embedding.embedding for embedding in embeddings.data]

    def encode_documents(self, documents: List[str]) -> List[np.array]:
        response = self._encode(documents)
        return [np.array(embedding) for embedding in response]

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        response = self._encode(queries)
        return [np.array(embedding) for embedding in response]

custom_embedding_instance = CustomEmbeddingFunction()

class VannaServer:
    def __init__(self, config):
        self.config = config
        self.vn = self._initialize_vn()
        self.embedding_function = custom_embedding_instance

    def _initialize_vn(self):
        config = self.config
        llm_type = config["llm_type"]
        model_ = config["model"]
        api_key = config["api_key"]
        ollama_host = config["ollama_host"] if "ollama_host" in config else None
        milvus_uri = config["milvus_uri"]
        sql_type = config["sql_type"]
        host = config["host"] if "host" in config else os.getenv("DB_HOST", "localhost")
        dbname = config["dbname"] if "dbname" in config else os.getenv("DB_NAME", "dify_data")
        user = config["user"] if "user" in config else os.getenv("DB_USER", "root")
        password = config["password"] if "password" in config else os.getenv("DB_PASSWORD", "mysql")
        port = config["port"] if "port" in config else int(os.getenv("DB_PORT", 3306))
        milvus_database = config["milvus_database"] if "milvus_database" in config else "test"
        milvus_client = MilvusClient(uri=milvus_uri,db_name=milvus_database)

        chat_llm = Ollama

        config = {
            'model': model_,  # 本地ollama大模型名称
            'api_key': api_key if api_key != '' else None,  # 本地ollama大模型服务地址
            'milvus_client': milvus_client,  # 本地milvus向量数据库服务地址
            "n_results": 12,
            "embedding_function": custom_embedding_instance,
        }

        if llm_type == "ollama":
            config['ollama_host'] = ollama_host # 本地ollama大模型服务地址

        elif llm_type == "tongyi":
            chat_llm = QianWenAI_Chat
        elif llm_type == "deepseek":
            chat_llm = DeepSeekChat

        MyVanna = make_vanna_class(ChatClass=chat_llm)
        vn = MyVanna(config)
        if sql_type == "postgres":
            vn.connect_to_postgres(host=host, dbname=dbname, user=user, password=password, port=port)
        elif sql_type == "mysql":
            vn.connect_to_mysql(host=host, dbname=dbname, user=user, password=password, port=port)

        if llm_type == "ollama":
            vn.client = SimpleNamespace(
                model=model_,
                api_key=api_key if api_key != '' else 'None',
                base_url=ollama_host + "/v1"
            )
        return vn

    def schema_train(self):
        # The information schema query may need some tweaking depending on your database. This is a good starting point.
        df_information_schema = self.vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS where table_schema = 'public'")

        # This will break up the information schema into bite-sized chunks that can be referenced by the LLM
        plan = self.vn.get_training_plan_generic(df_information_schema)
        # print(plan)

        # If you like the plan, then uncomment this and run it to train
        self.vn.train(plan=plan)

    def get_create_table_sql(self):
        sql = """
SELECT
    'CREATE TABLE '
    || C.TABLE_NAME
    || ' ('
    || C.COLUMN_NAMES
    || ');'
    || C.COMMENT_COLUMNS
		|| CASE WHEN FK.FOREIGN_KEY_COLUMNS IS NOT NULL THEN FK.FOREIGN_KEY_COLUMNS ELSE '' END
		|| CASE WHEN FK.FOREIGN_KEY_DESC IS NOT NULL THEN FK.FOREIGN_KEY_DESC ELSE '' END
    || 'COMMENT ON TABLE '
    || C.TABLE_NAME
    || ' IS '''
    || G.DESCRIPTION
    || ''';'
		AS DDL,
    C.TABLE_NAME
FROM (
    SELECT
        COL.TABLE_NAME,
        COL.TABLE_SCHEMA,
        STRING_AGG(
            COL.COLUMN_NAME
            || ' '
            || COL.DATA_TYPE
            || COALESCE('(' || COL.CHARACTER_MAXIMUM_LENGTH || ')', '')
            || COALESCE(' DEFAULT ' || COL.COLUMN_DEFAULT, '')
            || CASE
                WHEN COL.IS_NULLABLE = 'NO' THEN ' NOT NULL'
                ELSE ''
              END,
            ','
        ) AS COLUMN_NAMES,
        STRING_AGG(
            'COMMENT ON COLUMN '
            || COL.TABLE_NAME
            || '.'
            || COL.COLUMN_NAME
            || ' IS '''
            || PGD.DESCRIPTION
            || ''';',
            ''
        ) AS COMMENT_COLUMNS
    FROM
        PG_CATALOG.PG_STATIO_ALL_TABLES AS ST
    INNER JOIN
        PG_CATALOG.PG_DESCRIPTION AS PGD
        ON PGD.OBJOID = ST.RELID
    INNER JOIN
        INFORMATION_SCHEMA.COLUMNS AS COL
        ON (
            COL.TABLE_SCHEMA = ST.SCHEMANAME
            AND COL.TABLE_NAME = ST.RELNAME
            AND COL.ORDINAL_POSITION = PGD.OBJSUBID
        )
    WHERE
        COL.TABLE_SCHEMA = 'public'
    GROUP BY
        COL.TABLE_SCHEMA,
        COL.TABLE_NAME
) C
LEFT JOIN (
    SELECT
        N.NSPNAME AS SCHEMA_NAME,
        C.RELNAME AS TABLE_NAME,
        D.DESCRIPTION
    FROM
        PG_CATALOG.PG_DESCRIPTION D
    JOIN
        PG_CATALOG.PG_CLASS C
        ON C.OID = D.OBJOID
    JOIN
        PG_CATALOG.PG_NAMESPACE N
        ON N.OID = C.RELNAMESPACE
    WHERE
        C.RELKIND = 'r'
        AND D.OBJSUBID = 0
) G
ON G.SCHEMA_NAME = C.TABLE_SCHEMA
AND G.TABLE_NAME = C.TABLE_NAME
LEFT JOIN (
    SELECT rel_src.relname AS source_table,
        STRING_AGG(
            'ALTER TABLE '
            || rel_src.relname
            || ' ADD CONSTRAINT '
            || con.conname
            || ' FOREIGN KEY ('
            || att_src.attname
            || ') REFERENCES '
            || rel_tgt.relname
            || '('
            || att_tgt.attname
            || ');'
            ,
            ''
      ) AS FOREIGN_KEY_COLUMNS,
        STRING_AGG(
                'COMMENT ON CONSTRAINT  '
                || con.conname
                || ' ON '
                || rel_src.relname
                || ' IS '''
                || d.description
                || ''';',
                ''
        ) AS FOREIGN_KEY_DESC
    FROM
        pg_constraint con
        JOIN pg_class rel_src ON rel_src.oid = con.conrelid
        JOIN pg_class rel_tgt ON rel_tgt.oid = con.confrelid
        JOIN pg_attribute att_src ON att_src.attrelid = rel_src.oid AND att_src.attnum = ANY(con.conkey)
        JOIN pg_attribute att_tgt ON att_tgt.attrelid = rel_tgt.oid AND att_tgt.attnum = ANY(con.confkey)
        LEFT JOIN pg_description d ON d.objoid = con.oid
    WHERE
        con.contype = 'f'
    GROUP BY
        rel_src.relname
) FK ON FK.source_table = C.TABLE_NAME
WHERE C.TABLE_NAME NOT IN ('flyway_table_dict','flyway_table_classify','flyway_schema_history','flyway_synonyms','flyway_biz_synonym_source')
"""
        return sql

    def get_create_table_sql2(self):
        sql = """
SELECT
    'T:'
    || C.TABLE_NAME
		|| '['
		||
		G.DESCRIPTION
		|| ']'
    || ' ('
    || C.COLUMN_NAMES
    || ')'
		AS DDL,
    C.TABLE_NAME
FROM (
    SELECT
        COL.TABLE_NAME,
        COL.TABLE_SCHEMA,
        STRING_AGG(
            COL.COLUMN_NAME
            || ':'
						|| case when COL.DATA_TYPE = 'character varying' then 's'
							 when COL.DATA_TYPE = 'numeric' then 'i'
							 when COL.DATA_TYPE = 'timestamp without time zone' then 'dt'
							 else 's'
							 end
						|| ' ['
						|| PGD.DESCRIPTION
						|| ']',
            ','
        ) AS COLUMN_NAMES
    FROM
        PG_CATALOG.PG_STATIO_ALL_TABLES AS ST
    INNER JOIN
        PG_CATALOG.PG_DESCRIPTION AS PGD
        ON PGD.OBJOID = ST.RELID
    INNER JOIN
        INFORMATION_SCHEMA.COLUMNS AS COL
        ON (
            COL.TABLE_SCHEMA = ST.SCHEMANAME
            AND COL.TABLE_NAME = ST.RELNAME
            AND COL.ORDINAL_POSITION = PGD.OBJSUBID
        )
    WHERE
        COL.TABLE_SCHEMA = 'public'
    GROUP BY
        COL.TABLE_SCHEMA,
        COL.TABLE_NAME
) C
LEFT JOIN (
    SELECT
        N.NSPNAME AS SCHEMA_NAME,
        C.RELNAME AS TABLE_NAME,
        D.DESCRIPTION
    FROM
        PG_CATALOG.PG_DESCRIPTION D
    JOIN
        PG_CATALOG.PG_CLASS C
        ON C.OID = D.OBJOID
    JOIN
        PG_CATALOG.PG_NAMESPACE N
        ON N.OID = C.RELNAMESPACE
    WHERE
        C.RELKIND = 'r'
        AND D.OBJSUBID = 0
) G
ON G.SCHEMA_NAME = C.TABLE_SCHEMA
AND G.TABLE_NAME = C.TABLE_NAME
LEFT JOIN (
    SELECT rel_src.relname AS source_table,
        STRING_AGG(
            'ALTER TABLE '
            || rel_src.relname
            || ' ADD CONSTRAINT '
            || con.conname
            || ' FOREIGN KEY ('
            || att_src.attname
            || ') REFERENCES '
            || rel_tgt.relname
            || '('
            || att_tgt.attname
            || ');'
            ,
            ''
      ) AS FOREIGN_KEY_COLUMNS,
        STRING_AGG(
                'COMMENT ON CONSTRAINT  '
                || con.conname
                || ' ON '
                || rel_src.relname
                || ' IS '''
                || d.description
                || ''';',
                ''
        ) AS FOREIGN_KEY_DESC
    FROM
        pg_constraint con
        JOIN pg_class rel_src ON rel_src.oid = con.conrelid
        JOIN pg_class rel_tgt ON rel_tgt.oid = con.confrelid
        JOIN pg_attribute att_src ON att_src.attrelid = rel_src.oid AND att_src.attnum = ANY(con.conkey)
        JOIN pg_attribute att_tgt ON att_tgt.attrelid = rel_tgt.oid AND att_tgt.attnum = ANY(con.confkey)
        LEFT JOIN pg_description d ON d.objoid = con.oid
    WHERE
        con.contype = 'f'
    GROUP BY
        rel_src.relname
) FK ON FK.source_table = C.TABLE_NAME
WHERE C.TABLE_NAME NOT IN ('flyway_table_dict','flyway_table_classify','flyway_schema_history','flyway_synonyms','flyway_biz_synonym_source')
"""
        return sql

    # 更新建表DDL语句
    def refresh_create_table_ddl_train(self):

        # sql = self.get_create_table_sql()
        # 生成SQL
        sql = self.get_create_table_sql2()

        # The information schema query may need some tweaking depending on your database. This is a good starting point.
        c_table_ddl_list = self.vn.run_sql(sql)

        # 将 DataFrame 转换为字典列表
        c_table_ddl_records = c_table_ddl_list.to_dict(orient='records')

        exist_ddl_data = self.vn.milvus_client.query(
            collection_name="vannaddl",
            output_fields=["*"],
            limit=10000,
        )
        # exists_list = filter(lambda m: m["ddl"].startswith("CREATE TABLE "), exist_ddl_data)
        # remove_ids = [exist["id"] for exist in exists_list]
        remove_ids = [exist["id"] for exist in exist_ddl_data]
        if len(remove_ids) > 0:
            self.vn.milvus_client.delete(collection_name="vannaddl", ids=remove_ids)
        # import pdb; pdb.set_trace()
        for table_ddl in c_table_ddl_records:
            self.vn.train(ddl=table_ddl["ddl"])

        self.vn.milvus_client.refresh_load(collection_name="vannaddl")


    def refresh_schema_train(self):
        exist_doc_data = self.vn.milvus_client.query(
            collection_name="vannadoc",
            output_fields=["*"],
            limit=10000,
        )
        exists_list = filter(lambda m: m["doc"].startswith("The following columns are in the "), exist_doc_data)
        remove_ids = [exist["id"] for exist in exists_list]
        if len(remove_ids) > 0:
            self.vn.milvus_client.delete(collection_name="vannadoc", ids=remove_ids)
        self.schema_train()
        self.vn.milvus_client.refresh_load(collection_name="vannadoc")

    def update_schema_train_list(self,docs : list[str]):
        exist_doc_data = self.vn.milvus_client.query(
            collection_name="vannadoc",
            output_fields=["*"],
            limit=10000,
        )
        exists_list = filter(lambda m: not m["doc"].startswith("The following columns are in the "), exist_doc_data)
        remove_ids = [exist["id"] for exist in exists_list]
        if len(remove_ids) > 0:
            self.vn.milvus_client.delete(collection_name="vannadoc", ids=remove_ids)
        dict_docs = self.get_dict_docs()
        docs.extend(dict_docs)

        for doc in docs:
            self.vn.train(documentation=doc)
        # self.schema_train()
        self.vn.milvus_client.refresh_load(collection_name="vannadoc")

    def get_dict_docs(self) -> list[str]:
        dict_docs = []
        sql = "select id,table_name,column_name,column_remark,table_remark,dict_values from flyway_table_dict"
        c_table_dict_list = self.vn.run_sql(sql)
        # 将 DataFrame 转换为字典列表
        c_table_dict_records = c_table_dict_list.to_dict(orient='records')

        table_names = list(set(item['table_name'] for item in c_table_dict_records))

        grouped = defaultdict(list)
        for table_dict in c_table_dict_records:
            table_name = table_dict['table_name']  # 分组依据字段
            grouped[table_name].append(table_dict)

        grouped_dict = dict(grouped)

        for table_name in table_names:
            columns_list = grouped_dict[table_name]
            dict_values = ';'.join(f"字段:{item['column_remark']}({item['column_name']})的值:{item["dict_values"]}" for item in columns_list)
            column = columns_list[0]
            doc = f"{column["table_remark"]}表:{column["table_name"]},{dict_values}"
            dict_docs.append(doc)
        return dict_docs


    # 更新建表DDL语句
    def run_sql(self, sql : str, params : dict):

        # sql = self.get_create_table_sql()
        # 生成SQL
        sql = self.get_create_table_sql2()

        # The information schema query may need some tweaking depending on your database. This is a good starting point.
        c_table_ddl_list = self.vn.run_sql(sql)

        # 将 DataFrame 转换为字典列表
        c_table_ddl_records = c_table_ddl_list.to_dict(orient='records')

        exist_ddl_data = self.vn.milvus_client.query(
            collection_name="vannaddl",
            output_fields=["*"],
            limit=10000,
        )
        # exists_list = filter(lambda m: m["ddl"].startswith("CREATE TABLE "), exist_ddl_data)
        # remove_ids = [exist["id"] for exist in exists_list]
        remove_ids = [exist["id"] for exist in exist_ddl_data]
        if len(remove_ids) > 0:
            self.vn.milvus_client.delete(collection_name="vannaddl", ids=remove_ids)
        # import pdb; pdb.set_trace()
        for table_ddl in c_table_ddl_records:
            self.vn.train(ddl=table_ddl["ddl"])

        self.vn.milvus_client.refresh_load(collection_name="vannaddl")

    def vn_train(self, question="", sql="", documentation="", ddl=""):
        if question and sql:
            # 训练问答对
            self.vn.train(
                question=question,
                sql=sql
            )
        elif sql:
            # You can also add SQL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.
            self.vn.train(sql=sql)

        if documentation:
            # Sometimes you may want to add documentation about your business terminology or definitions.
            self.vn.train(documentation=documentation)

        if ddl:
            # You can also add DDL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.
            self.vn.train(ddl=ddl)

    def get_training_data(self):
        training_data = self.vn.get_training_data()
        # print(training_data)
        return training_data

    def ask(self, question, visualize=True, auto_train=True, *args, **kwargs):
        sql, df, fig = ask(self.vn, question, visualize=visualize, auto_train=auto_train, *args, **kwargs)
        return sql, df, fig

    def generate_sql(self, question,tenant_id:int, **kwargs):
        """
        Example:
        ```python
        vn.generate_sql("What are the top 10 customers by sales?")
        ```

        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:

        - [`get_similar_question_sql`][vanna.base.base.VannaBase.get_similar_question_sql]

        - [`get_related_ddl`][vanna.base.base.VannaBase.get_related_ddl]

        - [`get_related_documentation`][vanna.base.base.VannaBase.get_related_documentation]

        - [`get_sql_prompt`][vanna.base.base.VannaBase.get_sql_prompt]

        - [`submit_prompt`][vanna.base.base.VannaBase.submit_prompt]


        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None
        import time
        start_time0_0 = time.time()
        embeddings = self.vn.embedding_function.encode_queries([question])
        start_time0 = time.time()
        # embeddings = self.vn.normalizes(embeddings)
        print(f"embedding_function - 执行时间：{start_time0 - start_time0_0:.4f} 秒")
        # self.vn.milvus_client.flush(collection_name="vannasql")

        # import pdb; pdb.set_trace()
        question_sql_list = self.vn.get_similar_question_sql(embeddings, **kwargs)
        index_info = self.vn.milvus_client.describe_index("vannasql", "vector")
        print("index_info",index_info)
        start_time0_1 = time.time()
        print(f"get_similar_question_sql - 执行时间：{start_time0_1 - start_time0:.4f} 秒")
        # question_sql_list = question_sql_list[0:1]
        ddl_list = self.vn.get_related_ddl(embeddings, **kwargs)
        start_time0_2 = time.time()
        print(f"get_related_ddl 执行时间：{start_time0_2 - start_time0_1:.4f} 秒")
        # ddl_list= ddl_list[0:1]
        # self.filter_ddl_with_llm(ddl_list=ddl_list,question=question)
        # import pdb; pdb.set_trace()
        doc_list = self.vn.get_related_documentation(embeddings, **kwargs)
        # doc_list.append(f"所有主表查询，必须加上条件tenant_id = {tenant_id}")
        start_time1 = time.time()
        start_time0_3 = time.time()
        print(f"get_related_documentation执行时间：{start_time0_3 - start_time0_2:.4f} 秒")
        print(f"查询向量数据库执行时间：{start_time1 - start_time0:.4f} 秒")
        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            tenant_id=tenant_id,
            **kwargs,
        )

        start_time = time.time()
        print(f"get_sql_prompt 执行时间：{start_time - start_time1:.4f} 秒")
        self.vn.log(title="SQL Prompt", message=prompt)
        # import pdb; pdb.set_trace()
        llm_response = self.vn.submit_prompt(prompt, **kwargs)
        end_time = time.time()
        self.vn.log(title="LLM Response", message=llm_response)
        print(f"执行时间：{end_time - start_time:.4f} 秒")
        return self.vn.extract_sql(llm_response)
        # return self.vn.generate_sql(question=question)

    def get_api_info(self, question, **kwargs) -> dict:
        # 获取所有的问句
        funcs = self.vn.get_related_func(question=question)
        print(funcs)
        if len(funcs) == 0:
            return {}

        wanted_keys = {"description", "params", "id", "ext_prompt"}
        api_prompt_list = [{k: v for k, v in f.items() if k in wanted_keys} for f in funcs]

        client = UserClient()
        api_info = client.chat(question=question,api_list=api_prompt_list)
        if api_info:
            result = next((f for f in funcs if f.get("id") == api_info["id"]), None)
            api_info = {**result, **api_info}

        return api_info
        # # 将字典转换为 JSON 字符串
        # json_str = json.dumps(api_prompt_list, ensure_ascii=False)
        # prompt = self.get_prompt_1(json_str)
        # message_prompt = [self.vn.system_message(prompt)]
        # message_prompt.append(self.vn.user_message(question))
        # # 初始化 Ollama 的 OpenAI 接口客户端
        # client = OpenAI(
        #     base_url=self.vn.client.base_url,
        #     api_key=self.vn.client.api_key,
        # )
        # response = client.chat.completions.create(
        #     model= "qwen2.5-coder-32b-instruct",
        #     messages=message_prompt,
        #     temperature=0.2,
        #     max_tokens=8192,
        # )
        #
        # filtered_text = response.choices[0].message.content.strip()
        # if not filtered_text or filtered_text == "null":
        #     return {"id" : ""}
        #
        # cleaned_json_str = filtered_text.replace('```json', '').replace('```', '').strip()
        # print("-------------", cleaned_json_str)
        # cleaned_json_str = cleaned_json_str.strip().strip('`')
        # parsed_dict:dict = json.loads(cleaned_json_str)
        # result = next((f for f in funcs if f.get("id") == parsed_dict["id"]), None)
        # api_info = {**result, **parsed_dict}
        # print(api_info)
        # return api_info

    def get_run_text2api(self, question, tenant_id: int, **kwargs) -> dict:
        # 通过模型，匹配最相似的API信息，及参数
        api_info:dict = self.get_api_info(question=question)
        # 验证环节
        if "id" not in api_info or not api_info["id"]:
            return {
                "api_status" : 0
            }

        # 根据API信息，执行接口
        run_api_info = self.get_run_api(api_info=api_info,tenant_id=tenant_id)
        return run_api_info

    def get_run_api(self, api_info: dict, tenant_id: int, **kwargs) -> dict:
        type = api_info["type"]
        # 获取API 信息
        ext = api_info["ext"]
        url = api_info["url"]
        description = api_info["description"]
        params = api_info["params"]
        content = api_info["content"]
        if not ext:
            ext = {}
        if type == "sql":
            return {
                "url" : url,
                "api_status" : 1,
                "description" : description,
                "body" : {
                    "sql" : content,
                    "params" : params,
                    "tenantId" : tenant_id,
                    "ext" : ext
                }
            }
        else:
            return {
                "url" : url,
                "api_status" : 1,
                "description" : description,
                "body" : {
                    "content" : content,
                    "params" : params,
                    "tenantId" : tenant_id,
                    "ext" : ext
                }
            }

    def filter_ddl_with_llm(self, ddl_list, question, **kwargs):
        """
        使用本地 Ollama 大模型过滤与问题无关的 DDL 字段。

        Args:
            ddl_list (list[str]): 来自 get_related_ddl() 的表结构列表。
            question (str): 用户输入的问题。
            model_name (str): Ollama 中已加载的模型名称，如 llama3、mistral、codellama。

        Returns:
            list[str]: 过滤后的 DDL 字段列表（格式不变）。
        """

        # 初始化 Ollama 的 OpenAI 接口客户端
        client = OpenAI(
            base_url="http://wsd.wisdomidata.com:19042/v1",
            api_key="sk-ollama"  # 不验证，随便写
        )

        # 拼接原始 DDL 内容作为 prompt
        ddl_text = "\n".join(ddl_list)

        prompt = f"""
你是一个精通数据库建模的专家。
请根据以下问题，分析并筛选出与之相关的 DDL 字段。删除所有与问题无关的字段。
【问题】：
{question}
【DDL 表结构列表】：
{ddl_text}
【要求】：
- 输出保留原始 DDL 格式。
- 仅保留与问题直接相关的字段和表。
- 不添加任何解释或注释，只输出精简后的结构。
    """
        import pdb; pdb.set_trace()
        message_prompt = [self.vn.system_message(prompt)]
        message_prompt.append(self.vn.user_message(question))
        import time
        start_time = time.time()
        # llm_response = self.vn.submit_prompt(message_prompt, **kwargs)
        # self.vn.log(title="LLM Response", message=llm_response)
        # print(llm_response)
        # return llm_response
        # 调用 Ollama 本地模型
        response = client.chat.completions.create(
            model="deepseek-coder-v2",
            messages=message_prompt,
            temperature=0.2,
            max_tokens=14096,
        )
        end_time = time.time()
        print(f"执行时间：{end_time - start_time:.4f} 秒")
        # 获取返回文本
        filtered_text = response.choices[0].message.content.strip()
        # 按换行切割为列表（格式与原始 ddl_list 一致）
        return filtered_text.splitlines()

    def get_sql_prompt(
        self,
        initial_prompt : str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        tenant_id: int,
        **kwargs,
    ):
        """
        Example:
        ```python
        vn.get_sql_prompt(
            question="What are the top 10 customers by sales?",
            question_sql_list=[{"question": "What are the top 10 customers by sales?", "sql": "SELECT * FROM customers ORDER BY sales DESC LIMIT 10"}],
            ddl_list=["CREATE TABLE customers (id INT, name TEXT, sales DECIMAL)"],
            doc_list=["The customers table contains information about customers and their sales."],
        )

        ```

        This method is used to generate a prompt for the LLM to generate SQL.

        Args:
            question (str): The question to generate SQL for.
            question_sql_list (list): A list of questions and their corresponding SQL statements.
            ddl_list (list): A list of DDL statements.
            doc_list (list): A list of documentation.

        Returns:
            any: The prompt for the LLM to generate SQL.
        """

        if initial_prompt is None:
            initial_prompt = f"You are a {self.vn.dialect} expert. " + \
                             "Please help to generate a SQL query to answer the question. Your response should ONLY be based on the given context and follow the response guidelines and format instructions. "

        initial_prompt = self.vn.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.vn.max_tokens
        )

        if self.vn.static_documentation != "":
            doc_list.append(self.vn.static_documentation)

        initial_prompt = self.vn.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.vn.max_tokens
        )
        from datetime import date
        # 获取今天的日期
        today = date.today()

        initial_prompt += (
            "===Response Guidelines \n"
            "1. If the provided context is sufficient, please generate a valid SQL query without any explanations for the question. \n"
            "2. If the provided context is almost sufficient but requires knowledge of a specific string in a particular column, please generate an intermediate SQL query to find the distinct strings in that column. Prepend the query with a comment saying intermediate_sql \n"
            "3. If the provided context is insufficient, please explain why it can't be generated. \n"
            "4. Please use the most relevant table(s). \n"
            "5. If the question has been asked and answered before, please repeat the answer exactly as it was given before. \n"
            f"6. Ensure that the output SQL is {self.vn.dialect}-compliant and executable, and free of syntax errors. \n"
            # f"7. All main tables in SQL must add the condition tenant_id={tenant_id}. \n" #所有SQL的主表必须增加条件tenant_id=101
            # f"7. 所有生成的SQL必须增加条件tenant_id={tenant_id}. \n" #所有SQL的主表必须增加条件tenant_id=101
            f"6. 如果涉及当前日期，SQL使用：CURRENT_DATE作为条件判断，当前月份：EXTRACT(MONTH FROM CURRENT_DATE)，当前年： EXTRACT(YEAR FROM CURRENT_DATE) \n"
            f"7. 今天日期：{today}. \n"
            f"8. The generated SQL must specify the query fields and cannot directly use *. \n" #所有查询必须使用字段，不要使用*
            f"9. 所有查询的字段别名使用中文,日期或时间字段的格式默认使用YYYY-MM-DD. \n" #所有查询必须使用字段，不要使用*
            f"10. 查询时主表且有查询条件的表、汇总表、分组表如果有del字段，必须加上del=0，没有del字段的不要加del=0. \n" #所有查询必须使用字段，不要使用*
        )

        message_log = [self.vn.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is None:
                print("example is None")
            else:
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.vn.user_message(example["question"]))
                    message_log.append(self.vn.assistant_message(example["sql"]))

        question += f"（主表增加 tenant_id = {tenant_id}）"
        message_log.append(self.vn.user_message(question))

        return message_log

    def find_documents(self, **kwargs) -> dict:
        doc_data = self.vn.milvus_client.query(
            collection_name="vannadoc",
            output_fields=["*"],
            limit=10000,
        )
        result = []
        if doc_data is not None:
            result = [t['doc'] for t in doc_data]
        return result

    def run_sql(self, sql):
        return self.vn.run_sql(sql=sql)

    def training_data_export(self,text:str):

        filter = ""
        if text:
            filter = f'text like "%{text}%"'

        training_data = self.vn.milvus_client.query(
            collection_name="vannasql",
            output_fields=["*"],
            filter=filter,
            limit=10000,
        )
        result = []
        if training_data is not None:
            result = [{"question":t['text'], "sql": t['sql']} for t in training_data]

        return result

    def training_data_import(self, data_list):

        empty_items = list(filter(
            lambda item: item['question'] is None or item['question'] == "" or item['sql'] is None or item['sql'] == "",
            data_list
        ))

        if bool(empty_items):
            return True

        exist_doc_data = self.vn.milvus_client.query(
            collection_name="vannasql",
            output_fields=["*"],
            limit=10000,
        )
        data_texts = {t["question"]: t for t in data_list}

        if bool(exist_doc_data):
           remove_ids = [item["id"] for item in exist_doc_data if item['text'] in data_texts ]

           if bool(remove_ids):
               self.vn.milvus_client.delete(collection_name="vannasql", ids=remove_ids)

        for item in data_list:
            self.vn.train(
                question=item["question"],
                sql=item["sql"],
            )

        self.vn.milvus_client.refresh_load(collection_name="vannasql")

        return False


    def func_data_import(self, data_list):

        empty_items = list(filter(
            lambda item: item['url'] is None or item['url'] == "" or item['description'] is None or item['description'] == "",
            data_list
        ))

        if bool(empty_items):
            return True

        exist_func_data = self.vn.milvus_client.query(
            collection_name="vannafunc",
            output_fields=["*"],
            limit=10000,
        )
        data_texts = {t["description"]: t for t in data_list}

        if bool(exist_func_data):
            remove_ids = [item["id"] for item in exist_func_data if item['description'] in data_texts ]

            if bool(remove_ids):
                self.vn.milvus_client.delete(collection_name="vannafunc", ids=remove_ids)

        for item in data_list:
            self.vn.add_func(
                description=item["description"],
                url=item["url"],
                params=item["params"],
                ext=item["ext"],
                type=item["type"],
                content=item["content"],
                ext_prompt=item["ext_prompt"],
                tags=item["tags"],
                word_keys=item["word_keys"],
                synonym=item["synonym"],
                word=item["word"],
            )

        self.vn.milvus_client.refresh_load(collection_name="vannafunc")

        return False

def make_vanna_class(ChatClass=Ollama):
    class MyVanna(Milvus_VectorStore, ChatClass):
        def __init__(self, config=None):
            Milvus_VectorStore.__init__(self, config=config)
            ChatClass.__init__(self, config=config)
            self.embedding_function = custom_embedding_instance

        def is_sql_valid(self, sql: str) -> bool:
            # Your implementation here
            return False

        def generate_query_explanation(self, sql: str):
            my_prompt = [
                self.system_message("You are a helpful assistant that will explain a SQL query"),
                self.user_message("Explain this SQL query: " + sql),
            ]

            return self.submit_prompt(prompt=my_prompt)

        def get_related_ddl(self, embeddings: List[np.array], **kwargs) -> list:
            # import pdb; pdb.set_trace()
            search_params = {
                "metric_type": "COSINE",
                # "params": {"nprobe": 8},
            }
            # embeddings = self.embedding_function.encode_queries([question])

            res = self.milvus_client.search(
                collection_name="vannaddl",
                anns_field="vector",
                data=embeddings,
                limit=self.n_results,
                output_fields=["ddl"],
                search_params=search_params
            )
            res = res[0]
            list_ddl = []
            res = [r for r in res if r["distance"] >= 0.48]
            for doc in res:
                print(doc["distance"])
                list_ddl.append(doc["entity"]["ddl"])
            return list_ddl

        def get_related_documentation(self, embeddings: List[np.array], **kwargs) -> list:
            search_params = {
                "metric_type": "COSINE",
                # "params": {"nprobe": 8},
            }
            # embeddings = self.embedding_function.encode_queries([question])
            res = self.milvus_client.search(
                collection_name="vannadoc",
                anns_field="vector",
                data=embeddings,
                limit=self.n_results,
                output_fields=["doc"],
                search_params=search_params
            )
            res = res[0]

            list_doc = []
            for doc in res:
                print(doc["distance"])
                list_doc.append(doc["entity"]["doc"])
            return list_doc

        def get_related_func(self, question: str, **kwargs) -> list:
            search_params = {
                "metric_type": "COSINE",
                # "params": {"nprobe": 64},
            }

            embeddings = self.embedding_function.encode_queries([question])

            res = self.milvus_client.search(
                collection_name="vannafunc",
                anns_field="vector",
                data=embeddings,
                limit=10,
                output_fields=["url","description","params","ext","id","type","content","ext_prompt","word_keys", "synonym", "word","tags"],
                search_params=search_params
            )
            res = res[0]
            list_func = []
            for doc in res:
                print(doc["distance"])
                params = json.loads(doc["entity"]["params"])
                url = doc["entity"]["url"]
                description = doc["entity"]["description"]
                ext = doc["entity"]["ext"]
                type = doc["entity"]["type"]
                content = doc["entity"]["content"]
                ext_prompt = doc["entity"]["ext_prompt"]
                tags = doc["entity"]["tags"]
                word_keys = doc["entity"]["word_keys"]
                synonym = doc["entity"]["synonym"]
                word = doc["entity"]["word"]
                id = doc["entity"]["id"]
                list_func.append({
                    "id" : id,
                    "params" : params,
                    "url" : url,
                    "type" : type,
                    "content" : content,
                    "ext_prompt" : ext_prompt,
                    "word_keys" : word_keys,
                    "synonym" : synonym,
                    "word" : word,
                    "ext" : ext,
                    "tags" : tags,
                    "description" : description
                })
            return list_func

        def get_similar_question_sql(self, embeddings: List[np.array], **kwargs) -> list:
            search_params = {
                "metric_type": "COSINE",
                # "params": {"nprobe": 8},
            }
            import time
            start_time0 = time.time()
            # embeddings = self.embedding_function.encode_queries([question])
            start_time1 = time.time()
            # print(f"embedding_function - 执行时间：{start_time1 - start_time0:.4f} 秒")
            res = self.milvus_client.search(
                collection_name="vannasql",
                anns_field="vector",
                data=embeddings,
                limit=self.n_results,
                output_fields=["text", "sql"],
                search_params=search_params
            )
            res = res[0]
            list_sql = []
            res = [r for r in res if r["distance"] >= 0.5]
            for doc in res:
                dict = {}
                print(doc["distance"])
                dict["question"] = doc["entity"]["text"]
                dict["sql"] = doc["entity"]["sql"]
                list_sql.append(dict)
            return list_sql

        def normalizes(self, vectors):
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        def _create_collections(self):
            self._create_sql_collection("vannasql")
            self._create_ddl_collection("vannaddl")
            self._create_doc_collection("vannadoc")
            self._create_func_collection("vannafunc")
            self._create_table_config_collection("table_conf")

        def _create_doc_collection(self, name: str):
            if not self.milvus_client.has_collection(collection_name=name):
                vannadoc_schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=False,
                )
                vannadoc_schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
                vannadoc_schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)
                vannadoc_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._embedding_dim)

                vannadoc_index_params = self.milvus_client.prepare_index_params()
                vannadoc_index_params.add_index(
                    field_name="vector",
                    index_name="vector",
                    index_type="FLAT",
                    metric_type="COSINE",
                )
                self.milvus_client.create_collection(
                    collection_name=name,
                    schema=vannadoc_schema,
                    index_params=vannadoc_index_params,
                    consistency_level="Strong"
                )

        def _create_sql_collection(self, name: str):
            if not self.milvus_client.has_collection(collection_name=name):
                vannasql_schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=False,
                )
                vannasql_schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
                vannasql_schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
                vannasql_schema.add_field(field_name="sql", datatype=DataType.VARCHAR, max_length=65535)
                vannasql_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._embedding_dim)

                vannasql_index_params = self.milvus_client.prepare_index_params()
                vannasql_index_params.add_index(
                    field_name="vector",
                    index_name="vector",
                    index_type="FLAT",
                    metric_type="COSINE",
                    # metric_type="L2",
                )
                self.milvus_client.create_collection(
                    collection_name=name,
                    schema=vannasql_schema,
                    index_params=vannasql_index_params,
                    consistency_level="Strong"
                )

        def _create_ddl_collection(self, name: str):
            if not self.milvus_client.has_collection(collection_name=name):
                vannaddl_schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=False,
                )
                vannaddl_schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
                vannaddl_schema.add_field(field_name="ddl", datatype=DataType.VARCHAR, max_length=65535)
                vannaddl_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._embedding_dim)

                vannaddl_index_params = self.milvus_client.prepare_index_params()
                vannaddl_index_params.add_index(
                    field_name="vector",
                    index_name="vector",
                    index_type="FLAT",
                    metric_type="COSINE",
                    # metric_type="L2",
                )
                self.milvus_client.create_collection(
                    collection_name=name,
                    schema=vannaddl_schema,
                    index_params=vannaddl_index_params,
                    consistency_level="Strong"
                )


        def _create_table_config_collection(self, name: str):
            # import pdb; pdb.set_trace()
            if not self.milvus_client.has_collection(collection_name=name):
                vannafunc_schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=False,
                )
                vannafunc_schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
                vannafunc_schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="table", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._embedding_dim)

                vannafunc_index_params = self.milvus_client.prepare_index_params()
                vannafunc_index_params.add_index(
                    field_name="vector",
                    index_name="vector",
                    index_type="FLAT",
                    metric_type="COSINE",
                    # metric_type="L2",
                )
                self.milvus_client.create_collection(
                    collection_name=name,
                    schema=vannafunc_schema,
                    index_params=vannafunc_index_params,
                    consistency_level="Strong"
                )

        def _create_func_collection(self, name: str):
            # import pdb; pdb.set_trace()
            if not self.milvus_client.has_collection(collection_name=name):
                vannafunc_schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=False,
                )
                vannafunc_schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
                vannafunc_schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="params", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="ext", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="ext_prompt", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="tags", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="word_keys", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="synonym", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="word", datatype=DataType.VARCHAR, max_length=65535)
                vannafunc_schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._embedding_dim)

                vannafunc_index_params = self.milvus_client.prepare_index_params()
                vannafunc_index_params.add_index(
                    field_name="vector",
                    index_name="vector",
                    index_type="FLAT",
                    metric_type="COSINE",
                    # metric_type="L2",
                )
                self.milvus_client.create_collection(
                    collection_name=name,
                    schema=vannafunc_schema,
                    index_params=vannafunc_index_params,
                    consistency_level="Strong"
                )

        def add_func(self, description: str, url: str, params: str, ext : str, type : str,
                     content : str, ext_prompt : str, word_keys: str, synonym: str,
                     word: str, tags : str) -> str:
            if len(description) == 0:
                raise Exception("description can not be null")
            _id = str(uuid.uuid4()) + "-func"
            embedding = self.embedding_function.encode_queries([description])[0]
            self.milvus_client.insert(
                collection_name="vannafunc",
                data={
                    "id": _id,
                    "description": description,
                    "url" : url,
                    "params" : params,
                    "word_keys": word_keys,
                    "synonym": synonym,
                    "word": word,
                    "ext" : ext,
                    "type" : type,
                    "content" : content,
                    "ext_prompt" : ext_prompt,
                    "tags" : tags,
                    "vector": embedding
                }
            )
            return _id


        def add_table_conf(self, table_name:str, description: str) -> str:
            if len(table_name) == 0:
                raise Exception("table can not be null")
            _id = str(uuid.uuid4()) + "-table-conf"
            embedding = self.embedding_function.encode_queries([description])[0]
            self.milvus_client.insert(
                collection_name="table_conf",
                data={
                    "id": _id,
                    "description": description,
                    "table" : table_name,
                    "vector": embedding
                }
            )
            return _id

    return MyVanna


# 使用示例
if __name__ == '__main__':
    config = {"supplier": "GITEE"}
    server = VannaServer(config)
    # server.schema_train()
    server.ask("汇总每个类别的销售量和销售额, 并按照销售量进行降序排列")
