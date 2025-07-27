#!/usr/bin/node

/* server.js */
var http = require('http');

http.createServer(function (req, res) {
    res.writeHead(200, {'Content-Type': 'text/plain'});
    res.end('Hello World!');
}).listen(8080);


/* async.js */
const fs = require('fs');
const fsp = require('fs').promises;

async function make_dir(dir) {
    await fsp.mkdir(dir);
    console.log(`make_dir: ${dir}`);
}

async function populate_dir(dir) {
    var nums = [...Array(10).keys()]
    for (var num of nums) {
        var base = `file${num}`;
        var fp = await fsp.open(`${dir}/${base}`, 'w');
        await fp.write(`This is file ${num} in ${dir}\n`);
        console.log(`make_file: ${dir}/${base}`);
        await fp.close();
    }
}

async function build_dir(dir) {
    await make_dir(dir);
    await populate_dir(dir);
}

var N = 10;
var ns = [...Array(N).keys()];
for (var n of ns) {
    dir = `cupcake${n}`;
    build_dir(dir);
}


/* processes.js */
const os = require('os');
const fs = require('fs');
const process = require('process');
const child_process = require('child_process');
const { spawn } = require('node:child_process');

var procs = new Array();
var basenames = new Array('bin', 'share', 'lib', 'include', 'etc');

for (var basename of basenames) {
    var proc = spawn('find', [`/usr/${basename}`, '-type', 'f']);
    var cmd = proc.spawnargs.join(" ");
    /* closure to capture proc and cmd */
    ((proc, cmd) => {
        proc.stdout.on('data', (data) => {
            data.toString().split('\n').map(line => console.log(`${cmd}: (stdout): ${line}`))
            //console.log(`${cmd}: (stdout): ${data}`);
        });
    })(proc, cmd)
    proc.stderr.on('data', (data) => {
        console.error(`${cmd}: (stderr): ${data}`);
    });
    proc.on('close', (code) => {
        console.log(`${cmd}: child process exited with code ${code}`);
    });
    procs.push(proc);
}


/* classes.js */

const os = require('node:os');
const fs = require('node:fs/promises');

const foo = function(a) {
    return a**2;
}

class Rectangle {

    constructor(width, height) {
        this.width = width;
        this.height = height;
    }

    area() {
        return this.width * this.height;
    }
}

var b = foo(42);
console.log(b);

r = new Rectangle(4, 5);
console.log(r.area());
r.width = 8;
console.log(r.area());

const freemem = () => {
    bytes = os.freemem()
    gigs = bytes/(1024**3)
    return Math.round(gigs * 1e2) / 1e2
}
console.log(`Free memory: ${freemem()}GB`);
