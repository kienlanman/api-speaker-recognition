from flask import jsonify, request
from config import client
from app import app
from bson.objectid import ObjectId
import os

from src.model.densenet import DenseNet201, DenseNet121, DenseNet169
from src.service.TransferAudio import compareTwoVoice, convertBase64CafToWav, predictEmbedding
from src.utils import get_last_checkpoint_if_any
import src.constants as c
import logging

@app.route('/voice/search', methods=['GET'])
def search_voice_info():
    args = request.args
    formSearch = {}
    if(args.get('name') is not None) :
        formSearch['name'] = args['name'];
    # Select the collection
    collection = client['mongodb'].voice
    output = [];
    for p in collection.find(formSearch).sort("_id", -1) :
        output.append({"objectId": str(p["_id"]), "name": p['name'], "voice": p['voice']});
    return jsonify(output);

@app.route('/voice/save', methods=['POST'])
def addVoice():
    formRequest = request.form;
    formAdd = {};
    formAdd["name"] = formRequest.get("name");
    formAdd["voice"] = formRequest.get("voice");
    voice_wav = convertBase64CafToWav(formRequest.get("voice"));
    formAdd["voice_wav"] = voice_wav;
    collection = client['mongodb'].voice
    try:
     collection.insert_one(formAdd);
     return ("Add success");
    except:
        return ("An exception occurred")

@app.route('/voice/update', methods=['POST'])
def updateVoice():
    #get param
    formRequest = request.form;
    formUpdate = {};
    formUpdate["name"] = formRequest.get("name");
    formUpdate["voice"] = formRequest.get("voice");
    _id = ObjectId(formRequest.get("_id"))

    #make query update
    collection = client['mongodb'].voice
    myquery = {"_id": _id}
    newvalues = {"$set": formUpdate}

    try:
     collection.update_one(myquery, newvalues);
     return ("update success");
    except:
        return ("An exception occurred")

@app.route('/voice/delete/<id>', methods=['DELETE'])
def deleteVoice(id):
    collection = client['mongodb'].voice
    # xoa file npy
    try:
        os.remove("sample-npy/"+id+".npy")
    except Exception:
        pass
    # xoa file tren db
    try:
     collection.delete_one({'_id': ObjectId(id)});
     return ("Delete success");
    except:
        return ("An exception occurred")

global model
#checkpoint_densenet = "C:/Users/kienv/PycharmProjects/pro-api-speaker/src/check_point/densenet121_model178000_0.08254.h5"
#checkpoint_densenet = "C:/Users/kienv/PycharmProjects/pro-api-speaker/src/check_point/model_229800_1.14093.h5"
checkpoint_densenet = "C:/Users/kienv/PycharmProjects/pro-api-speaker/src/check_point/densenet169_model_131800_3.15211.h5"
model = DenseNet169(input_shape=(160, 64, 1) ,weights=checkpoint_densenet, include_top=False, classes=512, pooling='avg');

@app.route('/voice/send-voice', methods=['POST'])
def sendVoice():
    formData = request.form;
    try:
     base64Input = convertBase64CafToWav(formData.get('file'));
     embeddingInput = predictEmbedding(model, base64Input)

     collection = client['mongodb'].voice
     outputSample = [];
     for p in collection.find().limit(10):
         outputSample.append({"name": p["name"], "voice_wav": p["voice_wav"], "_id":str(p["_id"])});
     nameIdent = 'no name';
     maxRate = -2;

     for s in outputSample:
        rate = compareTwoVoice(model, embeddingInput, s)
        if maxRate < rate:
            maxRate = rate
            nameIdent = s['name']
     if maxRate < c.THRESHOLD:
         return "Không tìm thầy giọng phù hợp"
     return (nameIdent);
    except Exception as e:
        logging.error('Error at %s', 'division', exc_info=e)
        return ("An exception occurred")







