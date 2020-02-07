var {PythonShell} = require('./node_modules/python-shell');

var {PythonShell} = require('python-shell');
PythonShell.run('D:/Sotsuken/webapp/main.py', null, function (err) {
  if (err) throw err;
  console.log('finished');
});