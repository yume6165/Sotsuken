//console.log("頑張れ！！炭治郎！！頑張れ！！");

//pythonの実行
var {PythonShell} = require('./node_modules/python-shell');
var pyshell = new PythonShell('D:Sotsuken/webapp/main.py');
pyshell.send('D:\\Sotsuken\\webapp\\public\\sample\\*');
pyshell.send('D:\\Sotsuken\\webapp\\public\\input\\*');
pyshell.on('message', function(data){
	console.log(data);
	console.log('finish');
})
//console.log("pythonの実行切ってる");


//１:モジュールのロード
const http = require('http');
const fs = require('fs');
const csvSync = require('csv-parse/lib/sync');
var mysql = require('mysql');

var mime = {
  ".html": "text/html",
  ".css":  "text/css"
  // 読み取りたいMIMEタイプはここに追記
};

const server = http.createServer();
//const server = http.createServer((req, res)=>{
	//res.setHeaderでヘッダーの設定ができる
	//res.setHeader('Set-Cookie', 'last_accsess=' + Date.now()+ ';');
	//res.end();
//});
server.on('request', getURL);

//４:待ち受け開始
server.listen(3000);
console.log('Server running');


//データベース情報
var mysql = require('mysql');
var connection = mysql.createConnection({
	host:'localhost',
	user:'readhtml',
	password:'themis2020',
	port:3306,
	database:'themis'
});

connection.connect();


function getURL(req, res){
	var url = req.url;
	console.log(url);
	
	if('/' == url){
		//これが呼び出されるたびにcookieから相手を判断してユーザーを判別する
		var cookie = req.headers.cookie;//ここにcookieの情報が入る
		var user = {themis_user_id:'none', themis_session_id:'none'};//ユーザ情報の初期化		
		console.log("Cookie : " + cookie);//確認のためcookieを表示
		
		if(cookie == null){
			//res.setHeader('Set-Cookie', key= + value + ';');
			//一つずつ追加していくスタイル
			//res.setHeader('Set-Cookie', 'last_accsess=' + Date.now()+ ';');
			//ログインしていないとき
			res.setHeader('Set-Cookie', ['themis_user_id='+'guest;','themis_session_id='+'0;']);
		
		}else{
			//keyを取り出して辞書にする
			var keys = cookie.split('; ');
			//cookieをfor文で回してuser_idとsessionを回収
			//順番が前後する可能性があるので，あえて辞書型で収容したほうが良さそう
			for(var i=0; i < keys.length; i++){
				var tmp = keys[i].split('=');
				user[tmp[0]] = tmp[1];//ユーザの情報を入れた
			}
			console.log(user);
		}
		
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
	}else if('/css/style_login.css' == url)
		{
			fs.readFile('D:/Sotsuken/webapp/public/style_login.css', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/css'});
			res.write(data);
			res.end();
			console.log("get CSS!");
		});
	}else if('/js/jquery.min.js' == url)
		{
			fs.readFile('D:/Sotsuken/webapp/node_modules/jquery/dist/jquery.min.js', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/javascript'});
			res.write(data);
			res.end();
			console.log("get CSS!");
		});
	}else if(url.search('images') !== -1)
	{//画像の参照が来たら画像フォルダを指定して返したい
		res.writeHead(200, {'Content-Type': 'image/jpeg; charset=utf-8'});
		res.end(fs.readFileSync('D:/Sotsuken/webapp/public' +url, 'binary'), 'binary');	
		console.log("get Image");
		
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
		});
	}else if(url.search('/anken_data/') !== -1)
		{//まだ変更加えていない
			fs.readFile('D:/Sotsuken/webapp/public/'+url, 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/csv'});
			res.write(data);
			res.end();
			console.log("get CSV!");
		});
	}else if(url == '/anken_search')
		{
			//console.log("Hello");
			pyshell = new PythonShell('D:Sotsuken/webapp/search.py');
			pyshell.send('D:\\Sotsuken\\webapp\\public\\input\\*');
			pyshell.on('message', function(data){
			res.writeHead(200, {'Content-Type': 'text/plain'});
			res.write(data);
			console.log(data);
			res.end();
			});
			
	}else if(url == '/db_search')
		{//
			//console.log("Hello");
			pyshell = new PythonShell('D:Sotsuken/webapp/search.py');
			pyshell.send('D:\\Sotsuken\\webapp\\public\\sample\\*');
			pyshell.on('message', function(data){
			res.writeHead(200, {'Content-Type': 'text/plain; charset=utf-8'});
			res.write(data);
			console.log(data);
			res.end();
			});
			
	}else if(url == '/anken_up')
		{//
			console.log("POST!");
			var data = '';
			//POSTデータをうけとる
			req.on('data', function(chunk){data += chunk}).on('end', function(){
				console.log(data);
			//画像を'D:/Sotsuken/webapp/public/input/に保存
			
			
			fs.readFile('D:/Sotsuken/webapp/public/index.html', 'UTF-8',function(err, data){
				res.writeHead(200, {'Content-Type': 'text/html'});
				res.write(data);
				res.end();
			});
		});
	}else if(url == '/login')
		{
			
			fs.readFile('D:/Sotsuken/webapp/public/login.html', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/html'});
			res.write(data);
			res.end();
			console.log("get Login!")
			});
	}else if(url == '/css/login_bg')
		{
			fs.readFile('D:/Sotsuken/webapp/public/images/login_bg.jpg', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'image/jpeg'});
			res.write(data);
			res.end();
			console.log("get Login background image!")
			});
	}else if(url == '/create_new')
		{
			fs.readFile('D:/Sotsuken/webapp/public/create_new.html', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/html'});
			res.write(data);
			res.end();
			console.log("get Create New account resistance!")
			});
	}else if(url =='/create_user')//こっちがユーザを作る方
		{
			
			var body ='';
			req.on('data', function(chunk){
				body += chunk;
			});
			
			req.on('end', function(){//bodyにlogin_nameとlogin_passが入っている
				console.log(body);
				var user = body.split('&');
				//console.log(user[0]);
				var name = user[0].replace('login_name=', '');
				var pass = user[1].replace('login_pass=', '');
				//console.log("name: "+name+", pass: "+pass);
				
				//データベースに接続
				connection.query("insert into users set ?", {'name':name, 'pw':pass}, (err, rows, fields)=>{
					if(err) throw err;
					console.log(row);
				});
				//connection.end();
				fs.readFile('D:/Sotsuken/webapp/public/login.html', 'UTF-8',function(err, data){
				res.writeHead(200, {'Content-Type': 'text/html'});
				res.write(data);
				res.end();
				//console.log("get Login!");
				});
			});
	}else if(url =='/login_user')//こっちがログインする方
		{
			var body ='';
			//var login_flag = false;
			//var r_pass ='Hello';
			//var session_id ='0';//セッションに登録用のID
			//var session_name = "";//セッションに登録用のuser_name
			req.on('data', function(chunk){
				body += chunk;
			});
			
			req.on('end', function(){//bodyにlogin_nameとlogin_passが入っている
				//console.log(body);
				var user = body.split('&');
				//console.log(user[0]);
				var name = user[0].replace('login_name=', '');
				var pass = user[1].replace('login_pass=', '');
				//console.log("name: "+name+", pass: "+pass);
				
				//データベースに問い合わせ（非同期処理のため，Promiseを利用）
				loginCheck(name, pass, res)
				.then(function(data){
					if(data == true){//パスワードがあって入ればログインしてセッションを開始
						return sessionStart(name);
					}else{//パスワードが間違ってた場合はログインページを再度表示
						res.end();
					}
				});
				
			});
			
	}else if(url == '/logout')//ログアウトするときの処理
		{
			
			fs.readFile('D:/Sotsuken/webapp/public/login.html', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/html'});
			res.write(data);
			res.end();
			//console.log("get Login!")
			});
			
	}else if(url == '/note')//受け取ったメモをDBに書き込む
		{	
			//postからノートの情報を得る
			var notes = {note_title:'none', note:'none'};
			//ユーザ情報の入れ物
			var user = {themis_user_id:'none', themis_session_id:'none'};//ユーザ情報の初期化	
			var name = '';
			var record_id = '';
			
			getNote(req, notes)
			.then(function (data){
				notes = data;
				//console.log(notes);
				//console.log("start getUser: "+note);
				return getUser(req,user);
				})
			.then(function (data){
				name = data;
				//console.log("start recordData: "+name);
				return recordData(name, notes, record_id);
				})
			.then(function (data){
				//console.log("record_id: "+data);
				return readIndex(res, data);
			})
			
			
			
		}
}




//以下はサーバー内で使用される関数

//メモの記録に関する関数
function getNote(req, notes){
	return new Promise(function (resolve){
		req.on('data', function(chunk){
				var tmp = ''; 
				tmp += chunk;
				tmp = tmp.split('&');
				var note_title = tmp[0].replace('note_title=', '');
				note_title = note_title.replace(/\+/g,' ');
				var note = tmp[1].replace('note=', '');
				note = note.replace(/\+/g,' ');
				//console.log(note);
				notes['note_title'] = note_title;
				notes['note'] = note;
				resolve(notes);
			});
	})
}

function getUser(req, user){
	return new Promise(function(resolve){
		//cookieから情報を習得
		var cookie = req.headers.cookie;//ここにcookieの情報が入る
		var keys = cookie.split('; ');
		//cookieをfor文で回してuser_idとsessionを回収
		//順番が前後する可能性があるので，あえて辞書型で収容したほうが良さそう
		for(var i=0; i < keys.length; i++){
			var tmp = keys[i].split('=');
			user[tmp[0]] = tmp[1];//ユーザの情報を入れた
			//console.log(user);
		}
		//console.log(user['user_id']);
		resolve(user['user_id']);
	})
}

function recordData(name, notes,record_id){
	return new Promise(function(resolve){
		
		//データベースに接続(sessionに登録するのはログアウトするときだけでOK)
		//console.log(notes);
		connection.query("insert into record set ?", {'user':name,'memo_title':notes['note_title'], 'memo':notes['note']}, (err, rows, fields)=>{
			if(err) throw err;
			//console.log(rows.insertId);
			record_id = rows.insertId;
			resolve(record_id);
		});
	})
}

function readIndex(res, record_id){
	return new Promise(function(resolve){
		
		res.setHeader('Set-Cookie', ['themis_session_id='+record_id+';']);
		
		fs.readFile('D:/Sotsuken/webapp/public/index.html', 'UTF-8',function(err, data){
			res.writeHead(200, {'Content-Type': 'text/html'});
			res.write(data);
			res.end();
			console.log("Record note on DB");
			});
	})
}


//ログインに関する関数
function loginCheck(name, pass, res){
	return new Promise(function (resolve){
	//データベースにパスワードを問い合わせ
		connection.query("select pw from users where ?", {'name':name}, (err, rows, fields)=>{
			if(err) throw err;
			console.log(rows);
			//console.log(r_pass);
			var r_pass = rows[0].pw;
			//console.log(r_pass);
					
			if(pass == r_pass)//ログイン成功
			{
				console.log("Login Success!");
				//cookieに情報を載せる
				res.setHeader('Set-Cookie', ['themis_user_id='+name+';']);
						
				fs.readFile('D:/Sotsuken/webapp/public/index.html', 'UTF-8',function(err, data){
				res.writeHead(200, {'Content-Type': 'text/html'});
				res.write(data);
				//res.end();
				});
				resolve(true);
					
			}else//ログイン失敗
			{
				console.log("Miss your pass");
				fs.readFile('D:/Sotsuken/webapp/public/login.html', 'UTF-8',function(err, data){
				res.writeHead(200, {'Content-Type': 'text/html'});
				res.write(data);
				//res.end();
				});
				resolve(false);
			}
		});
	})
}


function sessionStart(name){//ユーザidを受け取ってセッション開始
	return new Promise(function(resolve){
		connection.query("insert into session set ?",{'user':name, 'status':true}, (err, rows, fields)=>{
						if(err) throw err;
						//console.log(rows);
						//セッションのidを受け取りたい
						var session_id = rows.insertId;
						var user_name = name;
						console.log("Welcome "+ name +"! Your session is , "+session_id+"!");
					
						//cookieに情報を載せる
						//res.setHeader('Set-Cookie', ['session_id='+session_id+';']);
					});
				
					res.end();
	})
}