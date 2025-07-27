#!/usr/bin/node

"use strict";

// core
const os = require('os');
const fs = require('fs');

// npm
const _ = require("underscore");

// local
module.paths.unshift(os.userInfo().homedir + '/lib/js');
const jlib = require('jlib');
var log    = jlib.log;
var log2   = jlib.log2;
var die    = jlib.die;
var colors = jlib.colors;

/* Below this is just an example of async IO in node.
 * Can delete everything below this if we're just using
 * this file as a module template.
 */

// Desired API:
// read(filename) -> {filename: ..., error: ..., data: ...}

var read = function(filename, label) {
    let o = {
        filename: filename,
        error: undefined,
        data:  undefined,
        label: label,
        // if we're using promises right, this should never really be needed
        done: false,
    };
    let promise = new Promise((resolve, reject) => {
        fs.readFile(filename, 'utf-8', (error, result) => {
            o.done = true;
            if (error) {
                o.error = error;
                reject(o);
            } else {
                o.data = result;
                resolve(o);
            }
        });
    });
    return promise;
}

/* EXAMPLE CODE: Here we read 5 files asynchronously, and then retrieve a 
 * specific subset of them to do a computation on the results. */

var promises = [
    read('/etc/conf.d/ntpd', 'gibberish'),
    read('/sys/class/backlight/intel_backlight/brightness', 'current'),
    read('/etc/conf.d/kexec', 'kexec'),
    read('/sys/class/backlight/intel_backlight/max_brightness', 'max'),
    read('/etc/conf.d/tor'),
    // read('/etc/shadow'), // to test rejection
];

let getByLabel = function(ps, label) {
    return ps.filter(v=>v.label===label).map(v=>parseFloat(v.data))[0]
};

/* using Promise.all to gather the results */
Promise.all(promises)
    .then(values => {
        let brightness     = getByLabel(values, 'current');
        let max_brightness = getByLabel(values, 'max');
        console.log(`Percentage is ${brightness / max_brightness}`);
    })
    .catch(values => {
        console.log("ERROR: At least one promise rejected:");
        console.log(values);
    });

console.log("Reached the end, but we're still aync!");

// one nice way to gather the results, if we don't want 
// to use the short-circuiting behavior of Promise.all.
/*
var wins  = [];
var fails = [];
promises.forEach((p, i) => {
    p.then(ret => {
        wins.push(ret);
    }).catch(ret => {
        fails.push(ret);
    })
    
});
*/
