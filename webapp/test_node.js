//var http = require("http"); 
//var fs = require('fs');
//httpとfsのモジュールを使います
//res.end(fs.createReadStream('D:/Sotsuken/webapp/public' +url), 'utf-8');	
	//	console.log("get Context");
	
'use strict'
const fs = require('fs');
const csv  = require('csv');
const csvSync  = require('csv-parse/lib/sync');

console.log(csvSync(fs.readFileSync('D:/Sotsuken/webapp/public/result/output_file/anken_dist/0_no_context.csv').toString()));