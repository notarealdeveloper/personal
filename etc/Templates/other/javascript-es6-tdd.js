/*
 * Run this with:
 * mocha es6-tests.js
 * 
 * To turn off the nearly invisible default colors:
 * mocha -C es6-tests.js
 */


// from http://tddbin.com/

// 21: spread - with-strings
// To do: make all tests pass, leave the assert lines unchanged!

const assert = require('assert')

describe('spread with strings', () => {

  it('simply spreads each char of a string', function() {
    const [a, b] = [...'ab'];
    assert.equal(a, 'a');
    assert.equal(b, 'b');
  });

  it('extracts each array item', function() {
    const [,a,b] = ['a', ...'12'];
    assert.equal(a, 1);
    assert.equal(b, 2);
  });
  
  it('works anywhere inside an array (must not be last)', function() {
    const letters = ['a', 'e', 'f', ...'bcd'];
    assert.equal(letters.length, 6);
  });
  
  it('dont confuse with the rest operator', function() {
    const [...rest] = [...'1234', '5'];
    assert.deepEqual(rest, [1, 2, 3, 4, 5]);
  });
  
  it('passed as function parameter', function() {
    const max = Math.max(...'12345');
    assert.deepEqual(max, 5);
  });
  
});

// 10: destructuring - array
// To do: make all tests pass, leave the assert lines unchanged!

describe('destructuring arrays makes shorter code', () => {

  it('extract value from array, e.g. extract 0 into x like so `let [x] = [0];`', () => {
    let [firstValue] = [1];
    assert.strictEqual(firstValue, 1);
  });

  it('swap two variables, in one operation', () => {
    let [x, y] = ['ax', 'why'];
    [x, y] = [y, x];
    assert.deepEqual([x, y], ['why', 'ax']);
  });
  
  it('leading commas', () => {
    const all = ['ax', 'why', 'zet'];
    const [,,z] = all;
    assert.equal(z, 'zet');
  });
  
  it('extract from nested arrays', () => {
    const user = [['Some', 'One'], 23];
    const [[firstName, surname], age] = user;
    
    const expected = 'Some One = 23 years';
    assert.equal(`${firstName} ${surname} = ${age} years`, expected);
  });

  it('chained assignments', () => {
    let c, d;
    let [a, b] = [c, d] = [1, 2];
    assert.deepEqual([a, b, c, d], [1, 2, 1, 2]);
  });

  it('in for-of loop', () => {
    for (var [, a, b] of [[0, 1, 2]]) {}
    assert.deepEqual([a, b], [1, 2]);
  });

});


// 11: destructuring - string
// To do: make all tests pass, leave the assert lines unchanged!

describe('destructuring also works on strings', () => {

  
  it('destructure every character', () => {
    let [a, b, c] = [...'abc'];
    assert.deepEqual([a, b, c], ['a', 'b', 'c']);
  });
  
  it('missing characters are undefined', () => {
    const [a,b,c] = 'ab';
    assert.equal(c, void 0);
  });
  
  it('unicode character work too', () => {
    const [, space, coffee] = [...'a ☕'];
    assert.equal(coffee, '\u{2615}');
  });
  
});


// 12: destructuring - object
// To do: make all tests pass, leave the assert lines unchanged!

describe('destructuring objects', () => {

  it('is simple', () => {
    const {x} = {x: 1};
    assert.equal(x, 1);
  });

  describe('nested', () => {
    it('multiple objects', () => {
      const magic = {first: 23, second: 42};
      const {magic: {second}} = {magic};
      assert.equal(second, 42);
    });
    it('object and array', () => {
      const {z:[,x]} = {z: [23, 42]};
      assert.equal(x, 42);
    });
    it('array and object', () => {
      const [,[{lang}]] = [null, [{env: 'browser', lang: 'ES6'}]];
      assert.equal(lang, 'ES6');
    });
  });
  
  describe('interesting', () => {
    it('missing refs become undefined', () => {
      const {z} = {x: 1, y: 2};
      assert.equal(z, void 0);
    });
  
    it('destructure from builtins (string)', () => {
      const {substr} = '1';
      assert.equal(substr, String.prototype.substr);
    });
  });

});


// 13: destructuring - defaults
// To do: make all tests pass, leave the assert lines unchanged!

describe('destructuring can also have default values', () => {

  it('for an empty array', () => {
    const [a=1] = [];
    assert.equal(a, 1);
  });

  it('for a missing value', () => {
    const [a,b=2,c] = [1,,3];
    assert.equal(b, 2);
  });

  it('in an object', () => {
    const {a, b=2} = {a: 1};
    assert.equal(b, 2);
  });

  it('if the value is undefined', () => {
    const {a, b=2} = {a: 1, b: void 0};
    assert.strictEqual(b, 2);
  });

  it('also a string works with defaults', () => {
    const [a, b=2] = '1';
    assert.equal(a, '1');
    assert.equal(b, 2);
  });

});


// 14: destructuring - parameters
// To do: make all tests pass, leave the assert lines unchanged!

describe('destructuring function parameters', () => {

  describe('destruct parameters', () => {
    it('multiple params from object', () => {
      const fn = ({id, name}) => {
        assert.equal(id, 42);
        assert.equal(name, 'Wolfram');
      };
      const user = {name: 'Wolfram', id: 42};
      fn(user);
    });
    
    it('multiple params from array/object', () => {
      const fn = ([{},{name}]) => {
        assert.equal(name, 'Alice');
      };
      const users = [{name: 'nobody'}, {name: 'Alice', id: 42}];
      fn(users);
    });
  });

  describe('default values', () => {
    it('for simple values', () => {
      const fn = (id, name='Bob') => {
        assert.strictEqual(id, 23);
        assert.strictEqual(name, 'Bob');
      };
      fn(23);
    });
    
    it('for a missing array value', () => {
      const defaultUser = {id: 23, name: 'Joe'};
      const fn = ([user = defaultUser]) => {
        assert.deepEqual(user, defaultUser);
      };
      fn([]);
    });
    
    it('mix of parameter types', () => {
      const fn = (id=1, [arr=2], {obj=3}) => {
        assert.equal(id, 1);
        assert.equal(arr, 2);
        assert.equal(obj, 3);
      };
      fn(void 0, [], {});
    });
  });

});


// 15: destructuring - assign
// To do: make all tests pass, leave the assert lines unchanged!

describe('assign object property values to new variables while destructuring', () => {

  describe('for simple objects', function() {
    it('use a colon after the property name, like so `propertyName: newName`', () => {
      const {x: y} = {x: 1};
      assert.equal(y, 1);
    });
    
    it('assign a new name and give it a default value using `= <default value>`', () => {
      const {x: y=42} = {z: 23};
      assert.equal(y, 42);
    });
    
    it('same as above, but different method', () => {
      const {a: b=2} = {a: 42};
      assert.equal(b, 42);
    });
  });
  
  describe('for function parameter names', function() {
    it('do it the same way, with a colon behind it', () => {
      const fn = ({x: y}) => {
        assert.equal(y, 1);
      };
      fn({x: 1});
    });
    
    it('giving it a default value is possible too, like above', () => {
      const fn = ({x: y=3}) => {
        assert.equal(y, 3);
      };
      fn({});
    });
  });
  
});


// 16: object-literal - computed properties
// To do: make all tests pass, leave the assert lines unchanged!

describe('Object literal properties may be computed values', () => {

  it('a computed property `x` needs to be surrounded by `[]`', () => {
    const propertyName = 'x';
    const obj = {[propertyName]: 1};
    assert.equal(obj.x, 1);
  });

  it('can also get a function assigned', () => {
    const key = 'func';
    const obj = {[key]: () => 'seven'};
    assert.equal(obj.func(), 'seven');
  });

  it('the key may also be the result of a function call', () => {
    const getName = () => 'propertyName';
    const obj = {[getName()]() {return 'seven'}};
    assert.equal(obj.propertyName(), 'seven');
  });

  it('the key can also be constructed by an expression', () => {
    const what = 'tyName';
    const obj = {['proper' + what]: null};
    assert.equal('propertyName' in obj, true);
  });

  it('accessor keys can be computed names too', () => {
    const obj = {
      get ['key']() {return 1}
    };
    assert.equal(obj.key, 1);
  });
});

