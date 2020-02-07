//pythonの実行
var {PythonShell} = require('./node_modules/python-shell');

var {PythonShell} = require('python-shell');
PythonShell.run('D:/Sotsuken/webapp/main.py', null, function (err) {
  if (err) throw err;
  console.log('finished');
});



//１:モジュールのロード
const http = require('http');
const fs = require('fs');
var mime = {
  ".html": "text/html",
  ".css":  "text/css"
  // 読み取りたいMIMEタイプはここに追記
};

const server = http.createServer();
server.on('request', getCss);

//４:待ち受け開始
server.listen(3000);
console.log('Server running');


function getCss(req, res){
	var url = req.url;
	if('/' == url){
		fs.readFile('D:/Sotsuken/webapp/public/index.html', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/html'});
			res.write(data);
			res.end();
			console.log("get HTML!")
			});
	}else if('/css/style.css' == url)
		{
			fs.readFile('D:/Sotsuken/webapp/public/style.css', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/css'});
			res.write(data);
			res.end();
			console.log("get CSS!")
		});
	}
	
}