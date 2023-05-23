# chatbot_demo
chatbot demo<br>
the demo contains graph description as text and only coruses information as text , i converted them also to embeddings <br>
the demo is using the dataset inside file resources\data.csv to generate prompt for gpt by comparing the similarity between the query embedding and the embeddings of dataset<br>
after downloading the project in your device<br>
apply the following commands inside the project folder :<br>
pip install virtualenv<br>
virtualenv venv<br>
venv\Scripts\activate <br>
(only if you an error appeared after applying this line ,execute the following command )<br>
set-executionpolicy remotesigned<br>
pip install Flask<br>
pip install requests<br>
<br>
run the server with following command :<br>
flask --app server.py  run -p 5000<br>
the demo needs openai api key , you can add it in consts.py
