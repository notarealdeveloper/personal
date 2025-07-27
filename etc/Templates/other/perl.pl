#!/usr/bin/perl

use feature ':5.10';
say "Hi there!";

print "This is a single statement.\n";
print "I'm ", "a ", "list.\n";
print 'Stuff in single quotes is taken literally, like in Bash.\n',"\n";

my $varnames = "variable names start with the \$ symbol.\n";
print "In perl, $varnames";

my $a = 5;
print "a = $a\n";
$a++;
print "a++ gives $a\n";
$a+=5;
print "a+=5 gives $a\n";
$a/=2;
print "a/=2 gives $a\n";

my $x = "8";
my $y = $x + "1";
my $z = $x . "1";
print "Adding the string $x to the string 1 with '+' gives: $y\n";
print "Adding the string $x to the string 1 with '.' gives: $z\n";
print "\n";

say '$ looks like "s", and it stands for "scalar"';
say '@ looks like "a", and it stands for "array"';

my @bag = (1, 2, 3, 4, 5, 6);
say "The first item in the bag is $bag[0]";

print "If an array value doesn't exist, perl will create it for you when you assign to it:\n";
my @winter_months = ("December", "January");
$winter_months[2] = "February";
print "$winter_months[2]\n";

say "If you want to figure out how many elements are in an array, assign the array to a scalar:";
my $nummonths = @winter_months;
say 'First, we type: my $nummonths = @winter_months';
say "say \$nummonths gives $nummonths";

say "Here's how to use hashes, also known as dictionaries: ";
my %days_in_month = (
    "January" => 31, "February" => 28.25, "April" => 30
);
$days_in_month{March} = 31;
say $days_in_month{February};
say $days_in_month{March};

# FOR LOOPS
for my $i (1, 2, 3, 4, 5) {
    say $i;
}

for my $i (1 .. 10) {
    say $i;
}

my @rangey = (1 .. 10);
my $maxval = 25;
for my $i (@rangey, 15, 20 .. $maxval) {
    say $i;
}

for my $marx ('Groucho', 'Harpo', 'Chico', 'Zeppo') {
    say "$marx is my favorite Marx brother.";
}

my %month_has = (1 => 31, 2 => 28.25, 4 => 30);
for my $i (keys %month_has) {
    say "$i has $month_has{$i} days.";
}

for my $i (keys %days_in_month) {
    say "$i has $days_in_month{$i} days.";
}

# REGULAR EXPRESSIONS

print "\n";

for my $phone ("392-9821", "200 4894", "(805) 452-7897") {
    say "$phone is a phone number" if $phone =~ /\d{3}[ -]\d{4}/;
}

print "\n";

for my $phone ("392-9821", "200 4894", "(805) 452-7897") {
    say "$phone is a strict phone number" if $phone =~ /^\d{3}[ -]\d{4}/;
}
