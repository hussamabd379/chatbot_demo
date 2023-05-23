from app.core.Controller import Controller
from app.core.Response import Response
import logging
import json
from app.consts import Consts
import openai
import pandas as pd  
import ast
from app.core.Utils import Utils
from app.models.Embedding import Embedding
from app.consts import Consts
class Chatbot(Controller):

    def generateEmbeddings(self):

       
        result = Embedding.generateEmbeddings()
        responseBody = json.dumps({
                'success': result,
        })
        return Response(body=responseBody)


    def ask(self,query):
        try:
            openai.api_key = Consts.OPENAI_KEY

            df = pd.read_csv(Consts.ROOT_DIR+"\\resources\\data.csv")
            df['embedding'] = df['embedding'].apply(ast.literal_eval)

            queryEmbedding = Embedding.getEmbeddings(Consts.EMBEDDING_MODEL, query)[0]

            strings_and_relatednesses = []
            for i, row in df.iterrows():
                relatedness = Utils.calculateRelatedness(queryEmbedding, row['embedding'])
                data = (row["text"], relatedness)
                strings_and_relatednesses.append(data)

 
            strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
            strings, relatednesses = zip(*strings_and_relatednesses)

            relatednessStrings = strings[:10]
            relatednessesValues = relatednesses[:10]
            
            token_budget: int = 4096 - 500

            information = "use the following data to answer the questions : \n"
            question = "question : " + query
            message = information
            i  = 0
            for relatednessString in relatednessStrings:
                data = "data section "+str(i)+" : \n "+relatednessString+" \n " 
                if (
                    Utils.num_tokens(message + data + question, model=Consts.GPT_MODEL)
                    > token_budget
                ):
                    break
                else:
                    message += data
                i = i+1

            messages = [
                {"role": "system", "content": "You are chatbot with name DooDoo for learning website"},
                {"role": "assistant", "content": message},
                {"role": "user", "content": question},
            ]
            response = openai.ChatCompletion.create(
                model=Consts.GPT_MODEL,
                messages=messages,
                temperature=0
            )

            realtedInfo = []
            for string,relatedness in zip(relatednessStrings, relatednessesValues):
                data = {}
                data["realtedness"] = relatedness
                data["string"] = string
                realtedInfo.append(data)
                

            response_message = response["choices"][0]["message"]["content"]
            responseBody = json.dumps({
                'success': True,
                'question': question,
                'answer' : response_message,
                'realted_info' : realtedInfo
            })
            return Response(body=responseBody)
        except Exception as inst:
            logging.error(inst)

