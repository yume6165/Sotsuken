//pythonの実行
var {PythonShell} = require('./node_modules/python-shell');
var pyshell = new PythonShell('D:Sotsuken/webapp/main.py');
pyshell.send('D:\\Sotsuken\\webapp\\public\\sample\\*');
pyshell.on('message', function(data){
	console.log(data);
	console.log('finish');
})


//１:モジュールのロード
const http = require('http');
const fs = require('fs');
const csvSync  = require('csv-parse/lib/sync');
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
			console.log("get CSS!");
		});
	}else if(url.search('/original_img/') !== -1)
	{//画像の参照が来たら画像フォルダを指定して返したい
		res.writeHead(200, {'Content-Type': 'image/jpeg; charset=utf-8'});
		res.end(fs.readFileSync('D:/Sotsuken/webapp/public' +url, 'binary'), 'binary');	
		console.log("get Original Image");
		
	}else if(url.search('/edge_img/') !== -1)
	{//画像の参照が来たら画像フォルダを指定して返したい
		res.writeHead(200, {'Content-Type': 'image/jpeg; charset=utf-8'});
		res.end(fs.readFileSync('D:/Sotsuken/webapp/public' +url, 'binary'), 'binary');	
		console.log("get Edge Image");
		
	}else if(url.search('/color_hist_img/') !== -1)
	{//画像の参照が来たら画像フォルダを指定して返したい
		res.writeHead(200, {'Content-Type': 'image/jpeg; charset=utf-8'});
		res.end(fs.readFileSync('D:/Sotsuken/webapp/public' +url, 'binary'), 'binary');	
		console.log("get Color Hist Image");
		
	}else if(url.search('/anken_dist/') !== -1)
		{
			fs.readFile('D:/Sotsuken/webapp/public/'+url, 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/csv'});
			res.write(data);
			res.end();
			console.log("get CSV!");
		});
	}
	
}