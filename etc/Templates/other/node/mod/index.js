console.log("mod imported")

function foo(x) {
    return x**2;
}

function bar(x) {
    return x+1;
}

module.exports = {
    foo: foo,
    bar: bar,
}
