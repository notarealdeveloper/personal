#include <iostream>

using namespace std;

/* 
 * Compile and run with:
 * f=filename.cpp && g++ -o $f{,.cpp} && ./$f
 * or
 * f=filename.cpp && clang++ -o $f{,.cpp} && ./$f
 */


/**************************/
/*** keyword parameters ***/
/**************************/

void kwargy(int x = 0, int y = 1)
{
    cout << "x = " << x << ", y = " << y << "\n";
}

void kwargs_example() {

    kwargy(6, 9);
    kwargy(6);
    kwargy();
}


/**********************/
/*** classes in c++ ***/
/**********************/

/* The syntax "Classname::funcname" tells the compiler
 * that the function is part of the class Classname. */

class Computer 
{
    public:
    Computer();     // Constructor
    ~Computer();    // Destructor
    void setspeed(int p);
    int  getspeed();

    protected:
    int processorspeed;
};

Computer::Computer()
{
    /* Constructors can accept arguments, but this one doesn't */
    processorspeed = 0;
}

Computer::~Computer()
{
    /* Destructors do not accept arguments. Makes sense. */
}

void Computer::setspeed(int p)
{
    processorspeed = p;
   /* This works too:
    * this->processorspeed = p; */
}

int Computer::getspeed()
{
    return processorspeed;
}

void class_example()
{
    Computer computer;
    computer.setspeed(100); 
    cout << computer.getspeed() << '\n';

   /* This only works if processorspeed is made public, as expected.
    * cout << computer.processorspeed << '\n'; */
}

/*****************************/
/*** abstract base classes ***/
/*****************************/

/* 
 * "Polygon" is the abstract base class,
 * of which Rectangle and Triangle are subclasses.
 * The term "polygon" is not particularly useful in itself,
 * but it is useful as a way to categorize more specific classes.
 */
class Polygon {
    protected:
        int width;
        int height;
    public:
        void set_values(int a, int b) {width = a; height = b;}
        virtual int area(void) = 0; /* default value, I guess */
};

class Rectangle: public Polygon {
    public:
    int area(void) {return (width * height);}
};

class Triangle: public Polygon {
    public:
    int area(void) {return (width * height / 2);}
};

void abc_example() {
    Rectangle rect;
    Polygon *poly1 = &rect;
    poly1->set_values(4, 5);

    Triangle trgl;
    Polygon *poly2 = &trgl;
    poly2->set_values(4, 5);

    cout << "Rect's area is: " << poly1->area() << endl;
    cout << "Trig's area is: " << poly2->area() << endl;
}

/**************************/
/*** function templates ***/
/**************************/

/* Templates let us write one function that operates similarly
 * on values of multiple different types. As such, templates are a kind 
 * of "polymorphic function."Their implementation in C++ requires
 * various black magic with the linker at compile time (or at least used
 * it did at one point). People often recommend against using them. */

template <typename T>  /* "template <class T>" works exactly the same */
T Max(T a, T b) {
    T result;
    result = (a > b) ? a : b;
    return result;
}

void templates_for_functions_example() {
    int i1 = 5, i2 = 6, imax;
    imax = Max<int>(i1, i2);
    cout << imax << endl;

    long l1 = 10, l2 = 5, lmax;
    lmax = Max<long>(l1, l2);
    cout << lmax << endl;

    float f1 = 8.672, f2 = 4.234, fmax;
    fmax = Max<float>(f1, f2);
    cout << fmax << endl;

    cout << "We can also call the functions without writing <type>:\n";
    cout << Max(i1, i2) << endl;
    cout << Max(l1, l2) << endl;
    cout << Max(f1, f2) << endl;
}


/***********************/
/*** class templates ***/
/***********************/

template <typename T> /* As always: same as "template <class T>" */
class Pair
{
    T a, b;
    public:
    Pair (T first, T second) {a=first; b=second;}
    T getmax();
    int getthree();
};

template <typename T>
T Pair<T>::getmax()
{
    T retval = (a > b) ? a : b;
    return retval;
}

void templates_for_classes_example()
{
    Pair <int> ints (100, 75);
    cout << ints.getmax() << endl;

    Pair <float> floats (74.123, 86.754);
    cout << floats.getmax() << endl;
}

/**********/
/*** io ***/
/**********/

#include <iostream>
#include <fstream>
#include <string>

void io_example() {
    ofstream myofstream;
    myofstream.open("example");
    myofstream << "I am the contents of the file.\n";
    myofstream.close();
    cout << "Alright, we wrote to the file. Now let's read from it.\n";
    cin.get();

    string line;
    ifstream myifstream("example");
    if (myifstream.is_open()) {
        while (!myifstream.eof()) {
            getline(myifstream, line);
            cout << line << endl;
        }
        myifstream.close();
    } else {
        cout << "Unable to open file";
    }
}

/******************/
/*** namespaces ***/
/******************/

/* In this case, there are two "global" variables with the same name, 
 * but with different types, and different initial values.
 * No name collisions occur thanks to the namespaces. */

namespace one
{
    int var = 5;
}

namespace two
{
    double var = 3.1416;
}

void ns_subexample_1()
{
    cout << one::var  << endl;
    cout << two::var << "\n\n";
}

void ns_subexample_2()
{
    /* The "using" keyword "includes" a namespace in the current scope 
     * So that we don't need to prepend the namespace to the variable */
    using namespace one;
    cout << var << endl;
    cout << two::var << "\n\n";
}

void ns_subexample_3()
{
    /* We can also define namespace aliases, and use "using" on them. */
    namespace ching = one;
    namespace chong = two;
    using namespace chong;
    cout << ching::var << endl;
    cout << var << "\n\n";
}

void namespace_example() 
{
    ns_subexample_1();
    ns_subexample_2();
    ns_subexample_3();
}

/*****************************/
/*** overloading functions ***/
/*****************************/

int show(int a)
{
    cout << "This is an int: " << a << "\n";
    return a;
}

float show(float a)
{
    cout << "This is a float: " << a << "\n";
    return a;
}

void overloading_functions_example()
{
    int   i = 7;
    float f = 8.0;
    show(i);
    show(f);
}


/*****************************/
/*** overloading operators ***/
/*****************************/

/* Notice that the class definiion includes the empty constructor 
 * (without parameters) and we have defined it with an empty block:
 * CVector () {};
 * This is needed, since we've explicitly declared another constructor:
 * CVector (int, int);
 * And when we explicitly declare any constructor, with any number of 
 * parameters, the default constructor with no parameters that the 
 * compiler can declare automatically is not declared, so we need to 
 * declare it ourselves in order to be able to construct objects of this 
 * type without parameters. Otherwise, the declaration "CVector c;"
 * below would not have been valid.
 * 
 * This all makes sense. It's similar to the fact that, in python,
 * changing the signature of a class's __init__ method from 
 * (self) to (self, arg) means we can't call the constructor with
 * zero arguments anymore, unless we make arg a keyword parameter.
 * In C++ the same idea is only slightly more clunky to express.
 */

class CVector {
    public:
    int x, y;
    CVector () {};
    // CVector () {x=0;y=0;};    /* This works too. */
    CVector (int, int);          /* A constructor for CVector class  */
    CVector operator+ (CVector); /* The + *method* acts like __add__ */
};

CVector::CVector (int a, int b) {
    x = a;
    y = b;
}


/* Read declaration something like this: 
 * Declaring a method called "operator+" of the CVector class, 
 * and this method takes a CVector parameter and returns a CVector.
 */
CVector CVector::operator+ (CVector that) {
    CVector sum;
    sum.x = x       + that.x;   /* "this" can be used implicitly... */
    sum.y = this->y + that.y;   /* ... or explicitly */
    return sum;
}

void overloading_operators_example() {
    CVector a(3, 1);
    CVector b(1, 2);
    CVector c = (a + b);
    /* This works too: CVector c = (a + b); */
    /* This works too: CVector c; c = a + b; */
    /* This works too: CVector c = a.operator+ (b); */
    cout << "c.x = " << c.x << '\n' << "c.y = " << c.y << '\n';
}

/*************************************/
/*** pass by value or by reference ***/
/*************************************/

void inc_pass_by_ref(int& x){x++;}
void inc_pass_by_val(int  x){x++;}

void pass_byval_or_byref_example() {
    int x = 7;
    inc_pass_by_val(x);
    cout << x << "\n";
    inc_pass_by_ref(x);
    cout << x << "\n";
}

/******************/
/*** exceptions ***/
/******************/

#define ZERO_DIVISION_BITCHES 22

void zerodivision_exception()
{
    try {
        int x;
        cout << "Gimmie a number. If it's zero I'll complain: ";
        cin >> x;
        if (!x)
            throw ZERO_DIVISION_BITCHES;    
        else
            cout << "Whew! One over your number is: " << 1.0/x << "\n\n";
    }
    catch (int e) {
        cout << "Bitch, don't divide by zero." << "\n\n";
    }
}

void academia_exception()
{
    try 
    {
        cout << "No matter what you do here, it will be wrong: ";
        cin.get();
        throw 666;
    }
    catch (int e) 
    {
        cout << "Exception number " << e << " occurred.\n\n";
    }
}

void fancier_exception()
{
    try {
        int x;
        char c = '7'; 
        int  i =  7;
        cout << "Press 1 for char. Press 2 for int: ";
        cin >> x;
        if      (x == 1)
            throw c;
        else if (x == 2)
            throw i;
        else
            throw "fuck";
    }
    catch (int param) {cout << "And we get an int exception.\n";}
    catch (char param) {cout << "And we get a char exception.\n";}
    catch (...) {cout << "Hey! We got a default exception.\n";}
}

void exceptions_example() {

    zerodivision_exception();
    academia_exception();
    fancier_exception();
}


int main ()
{
    kwargs_example();
    class_example();
    abc_example();
    templates_for_functions_example();
    templates_for_classes_example();
    io_example();
    namespace_example();
    overloading_functions_example();
    overloading_operators_example();
    pass_byval_or_byref_example();
    exceptions_example();
}
