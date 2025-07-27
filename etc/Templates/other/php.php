#!/usr/bin/php -f
<?php

// make a linked list of pages

/* array sorting
 * -------------
 * sort() - sort arrays in ascending order
 * rsort() - sort arrays in descending order
 * asort() - sort associative arrays in ascending order, according to the value
 * ksort() - sort associative arrays in ascending order, according to the key
 * arsort() - sort associative arrays in descending order, according to the value
 * krsort() - sort associative arrays in descending order, according to the key
 */

/* superglobals
 * -------------
 * $GLOBALS
 * $_SERVER
 * $_REQUEST
 * $_POST
 * $_GET
 * $_FILES
 * $_ENV
 * $_COOKIE
 * $_SESSION
 */

/* syntax
 * ------
 * PHP's is *not* case-sensitive, with the exception of variable names.
 */

/* define can make constants */
define("GREETING", "Welcome to this php script!", /* case_insensitive = */ true);
echo greeting . "\n";

function new_example() {
    echo "\n";
}

new_example();

/* classes */
class Object {

    public $name;
    public $number;
    public static $instances = 0;

    public function __construct($name) {
        $this->name = $name;
        $this->id = uniqid("", true);
        $this->number = self::$instances;
        self::$instances += 1;
    }

    public function rename($value) {
        $this->name = $value;
        return $this;
    }

    public function describe() {
        echo "object number {$this->number} has id {$this->id} and name {$this->name}" . PHP_EOL;
    }

}

$a = new Object("basket");
$b = new Object("hotdog");
$a->describe();
$b->describe();
$a->rename("cupcake");
$a->describe();

new_example();

/* switch */

$colors = array("red", "blue", "green", "purple");

// $rand_keys = array_rand($input, 2);
$random_index = array_rand($colors);
$color = $colors[$random_index];

switch ($color) {
    case "red":
    case "blue":
    case "green":
        echo "Your favorite color is $color!\n";
        break;
    default:
        echo "Your favorite color is a mystery!\n";
}

new_example();

/* c-style errors */

/* error handling */

function handle_errors($errno, $errstr) {

    echo "Error $errno: $errstr\n";

    switch ($errno) {
        case 2:
        case 1024:
            echo " * Fuck the police... continuing anyway\n";
            break;
        default:
            die(" * Dying gracefully\n");
    }
}

set_error_handler("handle_errors");

$your_moms_weight = 1/0;

trigger_error("The trigger_error function warns us of danger ahead");
trigger_error("Winter is coming...", E_USER_ERROR);

new_example();

/* exceptions */

exit(0);

/* foreach */

$values = array("yo", "mama", "so", "fat");
echo "Reasons why yo mama so fat:\n";
foreach ($values as $value) {
    echo " * $value" . PHP_EOL;
}


/* networking */

// use the curl extension to query google
$url = "https://www.google.com";
$ch = curl_init();
$timeout = 5;
curl_setopt($ch, CURLOPT_URL, $url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, $timeout);
$html = curl_exec($ch);
curl_close($ch);

# Create a DOM parser object
$dom = new DOMDocument();

# Parse the HTML from Google.
# The @ before the method call suppresses any warnings that
# loadHTML might throw because of invalid HTML in the page.
@$dom->loadHTML($html);

# iterate over all the <a> tags
foreach ($dom->getElementsByTagName('a') as $link) {

    echo $link->getAttribute('href') . PHP_EOL;

}

?>
