const express = require('express');
const app = express();
const PORT = process.env.PORT || 3000;
const http = require('http');
const fs = require('fs');
const csvSync = require('csv-parse/lib/sync');


app.listen(PORT);

app.post('/',(req, res) =>{
	let buffers = [];
	let cnt = 0;
	
	req.on('data', (chunk) => {
		buffers.push(chunk);
		comsole.log(++cnt);
	});
	
	req.on('end', () =>{
		console.log('[done] Image upload');
		req.rawBody = Buffer.concat(buffers);
		//書き込み
		fw.writeFile('./img.jpeg', req.rawBody, 'utf-8',(err) => {
			if(err) return;
			console.log('[done Image save]');
		})
	})
	
})

app.get('/', function (req, res) {
  fs.readFile('D:/Sotsuken/webapp/public/index.html', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/html'});
			res.write(data);
			res.end();
			console.log("get HTML!")
			});
});


app.get('/css/style.css', function (req, res) {
	fs.readFile('D:/Sotsuken/webapp/public/style.css', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/css'});
			res.write(data);
			res.end();
			console.log("get CSS!");
});

app.get('/js/jquery.min.js', function (req, res) {
	fs.readFile('D:/Sotsuken/webapp/node_modules/jquery/dist/jquery.min.js', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/javascript'});
			res.write(data);
			res.end();
			console.log("get CSS!");
});

app.get('/original_img', function (req, res) {
	var url = req.url;
	res.writeHead(200, {'Content-Type': 'image/jpeg; charset=utf-8'});
		res.end(fs.readFileSync('D:/Sotsuken/webapp/public' +url, 'binary'), 'binary');	
		console.log("get Original Image");
});

app.get('/edge_img', function (req, res) {
	var url = req.url;
	res.writeHead(200, {'Content-Type': 'image/jpeg; charset=utf-8'});
		res.end(fs.readFileSync('D:/Sotsuken/webapp/public' +url, 'binary'), 'binary');	
		console.log("get Edge Image");
});

app.get('/color_hist_img', function (req, res) {
	var url = req.url;
	res.writeHead(200, {'Content-Type': 'image/jpeg; charset=utf-8'});
		res.end(fs.readFileSync('D:/Sotsuken/webapp/public' +url, 'binary'), 'binary');	
		console.log("get Color Hist Image");
});

app.get('/anken_dist', function (req, res) {
	var url = req.url;
	fs.readFile('D:/Sotsuken/webapp/public/'+url, 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/csv'});
			res.write(data);
			res.end();
});

app.get('/anken_data', function (req, res) {
	var url = req.url;
	fs.readFile('D:/Sotsuken/webapp/public/'+url, 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/csv'});
			res.write(data);
			res.end();
			console.log("get CSV!");
});

app.get('/anken_search', function (req, res) {
			pyshell = new PythonShell('D:Sotsuken/webapp/search.py');
			pyshell.send('D:\\Sotsuken\\webapp\\public\\input\\*');
			pyshell.on('message', function(data){
			res.writeHead(200, {'Content-Type': 'text/plain'});
			res.write(data);
			console.log(data);
			res.end();
			});
});

app.get('/db_search', function (req, res) {
			pyshell = new PythonShell('D:Sotsuken/webapp/search.py');
			pyshell.send('D:\\Sotsuken\\webapp\\public\\sample\\*');
			pyshell.on('message', function(data){
			res.writeHead(200, {'Content-Type': 'text/plain'});
			res.write(data);
			console.log(data);
			res.end();
			});
});

app.get('/', function (req, res) {
	
});