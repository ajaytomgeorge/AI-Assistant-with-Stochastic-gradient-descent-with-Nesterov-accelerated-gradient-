from tkinter import *
import time
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('created_model.h5')
import json
import random
intents = json.loads(open('./data/intents.json').read())
words = pickle.load(open('./created_assets/created_words.pkl','rb'))
classes = pickle.load(open('./created_assets/created_classes.pkl','rb'))



        

def main():
    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(sentence, words, show_details=True):
        sentence_words = clean_up_sentence(sentence)
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        return(np.array(bag))

    def predict_class(sentence, model):
        p = bow(sentence, words,show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(msg):
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
        return res



    def send():
        msg = txtMsg.get('0.0', END)
        txtMsg.delete("0.0",END)

        if msg != '':
            txtMsgList.config(state=NORMAL)
            txtMsgList.insert(END, "You: " + msg + '\n\n')
            txtMsgList.config(foreground="#442265", font=("Verdana", 12 ))
        
            res = chatbot_response(msg)
            txtMsgList.insert(END, "Bot: " + res + '\n\n')
                
            txtMsgList.config(state=DISABLED)
            txtMsgList.yview(END)
        
        

    def sendMsg():#send messages
        strMsg = "I:" + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())+ '\n'
        txtMsgList.insert(END, strMsg, 'greencolor')
        txtMsgList.insert(END, txtMsg.get('0.0', END))
        # txtMsg.delete('0.0', END)
        send()

    def cancelMsg():#Cancel message
        txtMsg.delete('0.0', END)

    def sendMsgEvent(event):#Send message event
        if event.keysym =='Up':
            sendMsg()
    #Create window
    app = Tk()
    app.title('AI Assistant')

    #Create a frame container
    frmLT = Frame(width = 500, height = 320, bg = 'white')
    frmLC = Frame(width = 500, height = 150, bg = 'white')
    frmLB = Frame(width = 500, height = 30)
    frmRT = Frame(width = 420, height = 500)

    #Create control
    txtMsgList = Text(frmLT)
    txtMsgList.tag_config('greencolor',foreground = '#008C00')#Create tag
    txtMsg = Text(frmLC)
    txtMsg.bind("<KeyPress-Up>", sendMsgEvent)
    btnSend = Button(frmLB, text = 'send', width = 8, command = sendMsg)
    btnCancel =Button(frmLB, text = 'cancel', width = 8, command = cancelMsg)
    imgInfo = PhotoImage(file = "1.gif")
    lblImage = Label(frmRT, image = imgInfo)
    lblImage.image = imgInfo

    #Window layout
    frmLT.grid(row = 0, column = 0, columnspan = 2, padx = 1, pady = 3)
    frmLC.grid(row = 1, column = 0, columnspan = 2, padx = 1, pady = 3)
    frmLB.grid(row = 2, column = 0, columnspan = 2)
    frmRT.grid(row = 0, column = 2, rowspan = 3, padx =2, pady = 3)

    #Fixed size
    frmLT.grid_propagate(0)
    frmLC.grid_propagate(0)
    frmLB.grid_propagate(0)
    frmRT.grid_propagate(0)

    btnSend.grid(row = 2, column = 0)
    btnCancel.grid(row = 2, column = 1)
    lblImage.grid()
    txtMsgList.grid()
    txtMsg.grid()

    #Main event loop
    app.mainloop()

if  __name__ == "__main__":
    main()