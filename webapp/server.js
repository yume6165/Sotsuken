//pythonの実行
//var {PythonShell} = require('./node_modules/python-shell');
//var pyshell = new PythonShell('D:Sotsuken/webapp/main.py');
//pyshell.send('D:\\Sotsuken\\webapp\\public\\sample\\*');
//pyshell.send('D:\\Sotsuken\\webapp\\public\\input\\*');
//pyshell.on('message', function(data){
//	console.log(data);
	//console.log('finish');
//})
console.log("pythonの実行切ってるよ");

var express = require('express');
var app = express();

var listener = app.listen(3000, function(){
	
	console.log(listener.adress().port);
})