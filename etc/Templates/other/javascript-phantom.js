#!/usr/bin/phantomjs

"use strict";

console.log('Hello, world!');

var page = require('webpage').create();

page.open('http://www.sample.com', function() {
    page.includeJs("https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js", function() {
        page.evaluate(function() {
            $("button").click();
        });
        phantom.exit()
    });
});
