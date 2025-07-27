#!/usr/bin/perl6

# The name "Rakudo" for the Perl 6 compiler was first suggested by Damian Conway.
# "Rakudo" is short for "Rakuda-dō" (with a long 'o'; 駱駝道), which is Japanese for "Way of the Camel".
# "Rakudo" (with a short 'o'; 楽土) also means "paradise" in Japanese.

# To use rakudo-star instead of rakudo, make sure the following two directories are in your PATH,
# since I installed using the method from my 'gitty' script.
# /opt/rakudo-star-2017.01/bin
# /opt/rakudo-star-2017.01/share/perl6/site/bin

say 'hello world' and 'hello world'.say;

say #`(This is an embedded comment) "Omg embedded comments lol";

# values know their type
say 
  1.WHAT, 
  0.25.WHAT, 
  1.23e4.WHAT,
;

# string concatenation
say "s" ~ "laughter";

my $kebab-case-variable = 69;
say "kebab-case-variable == " ~ $kebab-case-variable ~ " (lol omg)";

# string coercion
say (~4.2).WHAT; # Str

# number coercion
say (+"42").WHAT; # Int

# arrays
my @countries = 'UK', 'Slovakia', 'Spain', 'Sweden';
say @countries.WHAT; # (Array)
say @countries.elems; # 4
say @countries[0]; # UK
@countries[4] = ’Czech Republic’; # lol unicode quotes work XD
say @countries.elems; # 5

@countries[1, 2] = @countries[2, 1];
say @countries;
say @countries[0, 1, 2];  # UK Spain Slovakia
say @countries[0..2];     # (the same)
say @countries[^3];       # (the same)
say @countries[*];        # prints all items
say @countries[];         # prints all items

# splits on whitespace, so we don't have to quote
my @spaceyArray = <This array has spaces>;
say @spaceyArray;

# splits on whitespace, but lets us insert
# elements that contain whitespace if we want.
my @spacierArray = << "This array" has spaces too >>;
say @spacierArray;

# you can push and pop on arrays
my @values;
@values.push(35);
@values.push(7);
@values.push(@values.pop + @values.pop);
say @values;

# hashes
my %capitals = Japan => 'Tokyo', China => 'Beijing';
say %capitals;

say %capitals{'Japan', 'China'}; # Tokyo Beijing

my %unicode-capitals = 日本 => '東京', 中國 => '北京';
say %unicode-capitals;

say %unicode-capitals{'日本', '中國'}; # Tokyo Beijing

say %unicode-capitals.keys;
say %unicode-capitals.values;

# The "given" statement

my $var = 42;
given $var {
    when 0..50 { say 'Less than or equal to 50'}
    when Int { say "is an Int" }
    when 42  { say 42 }
    default  { say "huh?" }
}

loop (my $i = 0; $i < 5; $i++) {
  say "The current number is $i"
}

# run: Runs an external command without involving a shell
# (seems to be an execl() wrapper)
run 'cowsay', 'I ran from perl6, nigga';

# shell Runs a command through the system shell.
# (seems to be a system() wrapper)
shell 'cowsay I did too bitch';

# Reading and writing files
my $fn = "/sys/class/backlight/intel_backlight/brightness";
my $data = slurp $fn;
say $data;
$data /= 2;
say $data;
spurt '/tmp/cake', $data;
if ($data == slurp '/tmp/cake') {
  say "File read/write worked!";
} 
shell "rm /tmp/cake";
say "Yes" if (5==6);

# integer types
say so :2<11111111> == 0b11111111 == :8<377> == 0o377 == 255 == 0d255 == :16<ff> == 0xff; # OUTPUT: «True␤»

# All forms allow underscores between any two 
# digits, which can serve as visual separators,
# but don't carry any meaning:
say 5_00000;       # five Lakhs 
say 500_000;       # five hundred thousand 
say 0xBEEF_CAFE;   # a strange place 
say :2<1010_1010>; # 0d170 

say :16("9F");         # 159 
say :100[99, 2, 3];    # 990203 (lol base 100)

say 68.888.round;

# interpolating arrays inside of strings
my @family = <Jason Sakura Debbie>;
say "The loser is @family[0]";
say "Top 3 were @family[]";

note 'note writes to stderr';

# Perl has generally followed the view that different semantics means
# different operator. This means that you know how the program
# will behave even if you don’t know the exact types of the data.
# There’s no question that == means numeric equality and eq means
# string equality, or that + is addition and ~ is concatenation.
# In Perl 6 we’ve de-confused a few things from Perl 5.
say "edam".flip;        # made
say (1, 2, 3).reverse;  # 3 2 1
say "omg" x 2;          # (string 'omgomg')
say "omg" xx 2;         # (a list containing 'omg' twice)
# ^^^
# Omg underloading is actually way nicer than overloading in some ways
# It steepens the learning curve at first, but flattens it for experts

my @dancers = <Jane Jill Jonny Jimmy Jenny Jack>;
for @dancers.pick(*) -> $first, $second {
  say "$first dances with $second";
}

# Ranges are objects
say 1 .. 5;     # 1, 2, 3, 4, 5
say 1 ^.. 5;    # 2, 3, 4, 5
say 1 ..^ 5;    # 1, 2, 3, 4
say 1 ^..^ 5;   # 2, 3, 4
say ^5;         # 0, 1, 2, 3, 4 (short for 0..^5)
for 0 ^.. 6 -> $a, $b {
  say "$a $b";
}
for (1 .. 9).pick(*) -> $a, $b, $c {
  say "$a $b $c";
}

# When we put a ? on a parameter, we make it optional.
# If no argument is passed for it, then it will contain Any.
# This "undefined" value is false in boolean context,
# so we can do this:
my @踊者 = <Jane Jill Jonny Jimmy Jenny>;
for @踊者.pick(*) -> $first, $second? {
  if $second {
    say "$first dances with $second";
  }
  else {
    say "$first dances alone";
  }
}

sub truncate($text, :$limit = 100, :$trailer = '...') {
  $text.chars > $limit
    ?? "$text.substr(0, $limit)$trailer"
    !! $text;
}
say truncate("Drink a beer", limit => 11); # Drink a bee...

# https://perl6advent.wordpress.com/2014/12/16/quoting-on-steroids/
my $a = 42;

say Q:s/foo $a/;   # foo 42    # / :: stfu syntax highlighting
say Q:b/foo $a/;   # foo $a    # / :: stfu syntax highlighting
say Q:s:b/foo $a/; # foo 42    # / :: stfu syntax highlighting

# short       long            what does it do
# =====       ====            ===============
# :q          :single         Interpolate \\, \q and \' (or whatever)
# :s          :scalar         Interpolate $ vars
# :a          :array          Interpolate @ vars
# :h          :hash           Interpolate % vars
# :f          :function       Interpolate & calls
# :c          :closure        Interpolate {...} expressions
# :b          :backslash      Interpolate \n, \t, etc. (implies :q)

my $pronoun = "guy";
my $pronouns = "guys";
my $func = -> {"gal"};
say Q:q/Well \hello there you's $pronouns./;   #/::
say Q:s/Well \hello there you's $pronouns./;   #/::
say Q:b/Well \hello there you's $pronouns./;   #/::
say Q:c/Well hello there you's {$pronoun}s./;  #/::
say Q:f/Well hello there you's &$func()s./;    #/::

my $o = 'e';
say Q:qq/"H{$o}y"/;

my @w  = Q:w/"What the" fuck is this/;
my @ww = Q:ww/"What the" fuck is this/;

say @w;
for @w -> $x {
  say $x;
}

say @ww;
for @ww -> $x {
  say $x;
}

say "Okay wait woah this next thing is cool...";
.say for qw/ foo bar 'first second' /;

# Proper heredoc indentation thank the fucking lord!
# The following text is "exdented" automatically by 
# the same number of indents as the target has.
my $world = 'there, friend!';
say qqto/MY_CUSTOM_EOF/;
  Hello $world
  How are you today?
  This is a goddamn proper heredoc.
  I love you Larry Wall. <3
  MY_CUSTOM_EOF

say <1 2 3> «+» <4 5 6>; # OUTPUT: «<5 7 9>␤»
say <1 2 3> «-» <4 5 6>; # OUTPUT: «<5 7 9>␤»
say <1 2 3> R«-» <4 5 6>; # OUTPUT: «<5 7 9>␤»
say <42 27 68> #`(NOT DONE YET!) <<+>> <27 42 1>; # OUTPUT: «<5 7 9>␤»
say "Holy lol that actually executed";
