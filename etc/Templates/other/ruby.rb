#!/usr/bin/ruby

# First impressions:
# Ruby is the bastard child of Perl and Python.
print "print is kinda like printf in C, or print in Perl\n"
puts  "puts is kinda like puts in C, or say in Perl"

# Defining functions
def conditionals(x)
    if x > 7
        print "Normal-ass function returned yep\n"
    else
        print "Normal-ass function returned nope\n"
    end
end

def one_liner_function(); puts "I'm a function, yo!"; end

conditionals(8)
conditionals(6)
one_liner_function()

# Closures and Lambdas and Scope: Example 1
def meta var
    proc { var + 1 }
end
f = meta(69)
puts f.call()

# Closures and Lambdas and Scope: Example 2
def metb(var)
    return lambda {|x| var + x }
end
g = metb(42)
puts g.call(5)

# Closures and Lambdas and Scope: Example 3
def metc var
    lambda {|x| var + x }
end
h = metb(100)
puts h.call(11)

# Command line arguments

# This is weird. I wonder how far this syntax goes?
if ARGV.first
    puts "ARGV.first == " + ARGV.first
end

# Unlike C, Python, Bash, etc., the 0th element isn't the program name.
if ARGV[0]
    puts "ARGV[0] == " + ARGV[0]
end

# unless and if: both can be one-lined
if     7 > 6; puts "Conditionals and loops can be made bashy";   end
unless 6 > 7; puts "The unless statement is just if in reverse"; end


# Regular expressions seem to work kinda like they do in Perl.
ip = "192.168.0.1"
regex = /([\d]{1,3}\.){3}[\d]{1,3}/
if ip =~ regex
    puts "It's an IP"
else
    puts "Not an IP"
end

num = "(910) 392-9821"
if num =~ /\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}/
    puts "It's a phone number"
else
    puts "Not a phone number"
end

# For loops don't require any prior declaration of the loop variable
for i in 0..3; puts "cake" ; end

# String operations
for i in 0..5; puts "Number = %s" % i ; end

puts "Ho " * 3

puts "Hello from " + self.to_s

a = "hello "
a << "world"
a.concat(33)
a.concat(0x21)
puts a

# This returns the index of the first match
puts "cat o' 9 tails" =~ /\d/
# This returns nil
puts nil == ("cat o' 9 tails" =~ 9)
puts "cat o' 9 tails" =~ 9

# Slices are pretty powerful
a = "hello there"
puts a[1]
puts a[2,6]
puts a[2..6]
puts a[/[aeiou](.)\1/]                      # Matches "ell"

# Regex slices can be "inlined"
str = "oo gixxif boop"
puts              str[/([aeiou])(.)\2\1/]   # Matches "ixxi"
puts "oo gixxif boop"[/([aeiou])(.)\2\1/]   # Matches "ixxi"

# How to import and use modules
require 'io/console'
rows, columns = $stdin.winsize
puts "Your screen is #{columns} columns wide and #{rows} rows tall"
