function typhoon(prompt,temp,model='typhoon-v1.5x-70b-instruct') {
  var url = 'https://api.opentyphoon.ai/v1/chat/completions';
  var headers = {
    'Authorization': 'typhoon_api_key',
    'Content-Type': 'application/json'
  };
  
  var payload = JSON.stringify({
    "model": model,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant. You must answer only in Thai and its dialect such as Isan langauge"
      },
      {
        "role": "user",
        "content": "".concat(prompt)
      }
    ],
    "max_tokens": 4096,
    "temperature": parseFloat(temp),
    "top_p": 0.7,
    "stream": false
});

  var options = {
    'method': 'post',
    'headers': headers,
    'payload': payload
  };
  
  var response = UrlFetchApp.fetch(url, options);
  var jsonResponse = JSON.parse(response.getContentText());
  var content = jsonResponse.choices[0].message.content;
    
  return content;
}

function chuemjit(prompt){
    var apiUrl = 'https://chuemjit-2024.et.r.appspot.com/api/query';
  var queryParams = {
    'query': prompt,
    'threshold': 0.8
  };

  var options = {
    'method' : 'post',
    'contentType': 'application/json',
    'payload' : JSON.stringify(queryParams)
  };

  try {
    var response = UrlFetchApp.fetch(apiUrl, options);
    var responseText = response.getContentText();
    Logger.log(responseText);
  } catch (e) {
    Logger.log('Error: ' + e.message);
  }
  return responseText
}

function getIsanTranslation(opt="Thai Script 1") {

  var textInput = "";

  // select output option
  if (opt == "Thai Script 1") {
    textInput = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet().getRange("B8").getValue();
  } else if (opt == "Thai Script 2") {
    textInput = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet().getRange("B9").getValue();
  } else if (opt == "Thai Script 3") {
    textInput = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet().getRange("B10").getValue();
  }

  var url = 'https://api.opentyphoon.ai/v1/chat/completions';
  var headers = {
    'Authorization': 'typhoon_api_key',
    'Content-Type': 'application/json'
  };
  
  var payload = JSON.stringify({
    "inputs": [
      {
        "role": "user",
        "content": "จงแปลประโยคที่จะให้เป็นภาษาอีสาน ".concat(textInput)
      }
    ],
    "max_tokens": 1000,
    "stop": ["<|eot_id|>", "<|end_of_text|>"],
    "model": "llama3-70b-typhoon"
  });

  var options = {
    'method': 'post',
    'headers': headers,
    'payload': payload
  };
  
  var response = UrlFetchApp.fetch(url, options);
  var contentLines = response.getContentText().split('\n')
  var jsonResponse = JSON.parse(contentLines[contentLines.length-3].substring(6));
    
  // Assuming the response contains the text you need in this structure
  var content = jsonResponse.completion;

  return content;
}

function getIsanVoice(textInputForVoice,token='',volume='1',speed='1',typeMedia='mp3',speaker='89') {
  if (token == '') {
    return "Please Input Voice Key in Cell B1 first";
  }
  var apiUrl = 'https://api-voice.botnoi.ai/api/service/generate_audio';
  var payload = {
    text: textInputForVoice,
    speaker: speaker.toString(),
    volume: parseFloat(volume),
    speed: parseFloat(speed),
    type_media: typeMedia
  };

  var options = {
    method: 'post',
    contentType: 'application/json',
    headers: {
      'Botnoi-Token': token
    },
    payload: JSON.stringify(payload)
  };

  var response = UrlFetchApp.fetch(apiUrl, options);
  var result = response.getContentText();

  var audioUrl = JSON.parse(result).audio_url;

  // Process the result as needed
  Logger.log(audioUrl);
  return audioUrl;
}