// modules in the current directory won't be found
// automatically, so we need to use relative paths
const mod = require('./mod');
console.log("Using mod functions");
console.log(mod.foo(4));
console.log(mod.bar(4));

// modules in ./node_modules will be found automatically
const mod2 = require('mod2');
console.log("Using mod2 functions");
console.log(mod2.foo2(4));
console.log(mod2.bar2(4));
