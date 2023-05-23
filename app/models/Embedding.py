from app.core.Model import Model
import openai
import logging
import pandas as pd  
import json
from app.consts import Consts

class Embedding(Model):
    @staticmethod
    def getEmbeddings(embeddingMode, text):

        response = openai.Embedding.create(model=embeddingMode, input=text)
        for i, be in enumerate(response["data"]):
            assert i == be["index"] 
        embedding = [e["embedding"] for e in response["data"]]
        return embedding
    
    @staticmethod
    def generateEmbeddings():
        openai.api_key = Consts.OPENAI_KEY
    
        with open( Consts.ROOT_DIR+"\\resources\\dataset.json", encoding='utf-8', errors='ignore') as f:
            contents = f.readlines()

        texts = []
        embeddings = []
        
        recommendationReasons = 'Recommnedation reasons for journey "Demenz als Herausforderung":'\
        +'Ihnen wird der Lernpfad empfohlen, weil es ersichtlich ist, dass die Themen in der Liste in Bezug auf "Demenz als Herausforderung" thematisch geordnet sind. Zunächst beschäftigen sich die ersten beiden Themen mit dem Konzept der Autonomie in Bezug auf Demenz. Hier wird erörtert, wie freiheitseinschränkende Maßnahmen, Digitalisierung und das Herausfordernde Verhalten von Demenzpatienten die Autonomie beeinträchtigen können. Danach befasst sich die nächste Themenreihe mit dem Zwang in Bezug auf Sozialisation und Entwicklungsaufgaben. Hier geht es darum, wie Gefängnisse und die Strafphilosophie den Zwang in der Sozialisation beeinflussen können, sowie wie Konflikte und Aggressionen ohne Druck und Kindererziehung gelöst werden können. Schließlich beschäftigen sich die letzten beiden Themen mit der Pathogenese und den Entwicklungsaufgaben im Kontext der Sozialisation. Hier wird erörtert, wie Validation und Alterstheorien die Pathogenese beeinflussen können und wie das Konzept des Goffmanschen Theaters und die Anlage-Umwelt-Debatte die Entwicklungsaufgaben beeinflussen können. Insgesamt zeigt die Themenordnung eine logische Struktur, die das Konzept von "Demenz als Herausforderung" in Bezug auf Autonomie, Zwang, Pathogenese und Entwicklungsaufgaben untersucht.'
        texts.append(recommendationReasons)
        
        graphDescription = 'you have the following text which represent graph of nodes'+". \n "\
                            +'(example : 1 is node 1.1 is a the first subnode of the node 1 and so on)'+". \n "\
                            +'1-Demenz als Herausforderung (journey)'+". \n "\
                            +'1.1-Grundlagen Demenz (course)'+". \n "\
                            +'1.1.1-Grundlegende Information zur Erkrankung Demenz (topic)'+". \n "\
                            +'1.1.2-Demenz als Herausforderung in der Häuslichen Pflege (topic)'+". \n "\
                            +'1.2-Komplex Autonomie (course)'+". \n "\
                            +'1.2.1-Autonomie-Zwang (topic)'+". \n "\
                            +'1.2.2-Autonomie - Salutogenese (topic)'+". \n "\
                            +'1.2.3-Autonomie - Pathogenese (topic)'+". \n "\
                            +'1.2.4-Autonomie - Inklusion (topic)'+". \n "\
                            +'1.2.5-Autonomie - Validation (topic)'+". \n "\
                            +'1.2.6-Autonomie - Entwicklungsaufgaben (topic)'+". \n "\
                            +'1.3-Komplex Zwang (course)'+". \n "\
                            +'1.3.1-Zwang - Sozialisation (topic)'+". \n "\
                            +'1.3.2-Zwang - Pathogenese (topic)'+". \n "\
                            +'1.3.3-Zwang - Institution (topic)'+". \n "\
                            +'1.3.4-Zwang - Entwicklungsaufgaben (topic)'+". \n "\
                            +'1.3.5-Pathogenese - Validation (topic)+". \n "'
        
        texts.append(graphDescription)

        try: 
            coursesNodes = json.loads(''.join(contents))
            coursesText = []

            for courseNode in coursesNodes:
                logging.error(courseNode)
                courseText = ""
                courseText += "title: " +"\n "
                courseText += courseNode["title"] +". \n "
                courseText += "type: " +"\n "
                courseText += courseNode["node_type"] +". \n "
                courseText += "description: " +"\n "
                courseText += courseNode["description"] +". \n "
                courseText += "explanation 1: " +"\n "
                courseText += courseNode["automatic_explanation"] +". \n "
                courseText += "explanation 2: " +"\n "
                courseText += courseNode["expert_explanation"] +". \n "
                courseText += "explanation 3: " +"\n "
                courseText += courseNode["expert_keywords"] +". \n "
                coursesText.append(courseText);
                
            
            texts.extend(coursesText)


            logging.error(coursesText[0])
            for text in texts:
                textEmbedding = Embedding.getEmbeddings(Consts.EMBEDDING_MODEL, text)
                embeddings.extend(textEmbedding)


            df = pd.DataFrame({"text": texts, "embedding": embeddings})
            SAVE_PATH = "D:\projects\python\similarity\data2.csv"
            df.to_csv(SAVE_PATH, index=False)

            return True
        
        except Exception as inst:
            logging.error(inst)
            return False
        