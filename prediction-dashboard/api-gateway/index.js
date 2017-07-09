var express = require('express');
var app = express();
var http = require('http')

// app Dependencies
var dep = {
	path: require('path'),
}

var Language = require('@google-cloud/language');
var projectId = 'process.env.GCP_PROJECT';
//key-decorator-169320
// Instantiates a client
var language = Language({
	projectId: projectId
});

// Use the Natural Language Prediction API
var predict = function(text, callback) {
  // Detects the sentiment of the text
  language.detectSentiment(text)
    .then((results) => {
      callback(results[0]);
    })
    .catch((err) => {
      console.error('ERROR:', err);
    });
}

// Use Express
app.use(express.static('./Client/'))

// Handles Users
app.get('/', function(req, res) {
	res.sendFile(__dirname + '/index.html');
}); 

// Handles Users
app.get('/:text', function(req, res) {
	var text = req.params.text
	predict(text, function (result) {
		console.log(result.score)
		res.json(result.score);
	});
});

// app Start Function
var port = 3000
app.listen(port);
console.log("app listening on port %d", port);
