use std::fmt;

#[allow(dead_code)]
fn main() {

    let x: i32 = 42;
    println!("x is {}", x);

    let array: [i32; 4] = [6,7,8,9];
    println!("array is {:?}", array);

    let s: &str = "hello world";
    println!("We start with: {s}");

    let s2: String = s.to_string();
    println!("We continue with: {s2}");

    // mutable vectors
    let mut vector: Vec<i32> = vec![4,5,6,7];
    vector.push(8);
    let slice: &[i32] = &vector;
    println!("vector: {:?} and slice: {:?}", vector, slice);

    // destructuring let
    let x: (i32, &str, f64) = (1, "hello", 3.4);
    let (a, b, c) = x;
    println!("Tuple destructured to: (a={}, b={}, c={})", a, b, c);

    ///////////
    // Types //
    ///////////

    // data Point = Point {x::Int, y::Int}
    struct Point {
        x: i32,
        y: i32,
    }

    // instance Display Point where
    //     fmt self stream = write stream string
    //          where string = ...
    impl fmt::Display for Point {
        // To use the `{}` marker, the trait `fmt::Display`
        // must be implemented manually for the type.
        // This trait requires `fmt` with this exact signature.
        fn fmt(&self, stream: &mut fmt::Formatter) -> fmt::Result {
            // Write strictly the first element into the supplied
            // output stream: `f`.
            // Returns `fmt::Result` which indicates whether the
            // operation succeeded or failed. Note that `write!`
            // uses syntax which is very similar to `println!`.
            return write!(stream, "Point(x={}, y={})", self.x, self.y);
        }
    }

    let origin: Point = Point {x: 0, y: 0};
    println!("impl fmt::Display shows: {}", origin);

    // data Point2 = Point2 Int Int
    // a struct with unnamed fields, called a "tuple struct"
    struct Point2(i32, i32);
    let origin2 = Point2(0, 0);
    println!("Point2 has x={} and y={}", origin2.0, origin2.1);

    // data Direction = Left | Right | Up | Down
    //      deriving (Show)
    // enum: basic C-style
    #[derive(Debug)]
    enum Direction {
        Left,
        Right,
        Up,
        Down,
    }

    let direction = Direction::Up;
    println!("direction is: {:?}", direction);

    // data OptionalI32 = AnI32 Int | Nothing
    // enum: with fields
    #[derive(Debug)]
    enum OptionalI32 {
        AnI32(i32),
        Nothing,
    }

    let two: OptionalI32 = OptionalI32::AnI32(2);
    let nah: OptionalI32 = OptionalI32::Nothing;
    println!("OptionalI32: {:?}", two);
    println!("OptionalI32: {:?}", nah);

    // data Maybe a = Just a | Nothing
    // generics
    #[derive(Debug)]
    enum Maybe<T> {
        Just(T),
        Nothing,
    }

    let x : Maybe<i32> = Maybe::Just(42);
    let y : Maybe<i32> = Maybe::Nothing;
    println!("Maybe: x is {:?} and y is {:?}", x, y);

    // Methods on structs
    struct Foo<T> {
        bar: T
    }

    impl<T> Foo<T> {
        fn bar(&self) -> &T {
            &self.bar
        }
        fn bar_mut(&mut self) -> &mut T {
            // self is mutably borrowed
            &mut self.bar
        }
        fn into_bar(self) -> T {
            // here self is consumed
            self.bar
        }
    }

    let afoo = Foo {bar: 42};
    let mut atoo = Foo {bar: 69};
    println!("afoo.bar(): {}", afoo.bar());
    println!("atoo.bar_mut(): {}", atoo.bar_mut());

    // linearity!
    //
    // can do this:
    println!("atoo.into_bar(): {}", atoo.into_bar());
    //
    // but can't do it twice ;)
    //
    // ERROR:
    //println!("atoo.into_bar(): {}", atoo.into_bar());

    // traits are typeclasses!
    trait Frobnicate<T> {
        fn frobnicate(self) -> Option<T>;
    }

    // instance Frobnicate Foo where
    impl<T> Frobnicate<T> for Foo<T> {
        fn frobnicate(self) -> Option<T> {
            Some(self.bar)
        }
    }

    println!("afoo.frobnicate() gives {:?}", afoo.frobnicate());

    //////////////////////
    // pattern matching //
    //////////////////////

    let foo : Option<i32> = Option::Some(42);
    let bar : Option<i32> = Option::None;
    let objs : [Option<i32>; 2] = [foo, bar];
    for obj in objs {
        match obj {
            Option::Some(n) => println!("it's a value: {}", n),
            Option::None    => println!("it's none"),
        }
    }


    //////////////////
    // linear logic //
    //////////////////


}
