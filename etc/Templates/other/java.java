class CHANGEME {

    /*
     * Compile and run with:
     * javac CHANGEME.java && java CHANGEME "yo mama" "so fat"
     */

    public static void main(String[] args) {
        /* Static method invocation */
        CHANGEME.chars();

        CHANGEME j = new CHANGEME();
        j.calling_other_classes();

        /* Accessing command line arguments */
        if (args.length > 0) {
            for (int i = 0; i < args.length; i++) {
                System.out.println("arg[" + i + "] = " + args[i]);
            }
        }

        /* String formatting */
        System.out.printf("Yo %s weigh %d pounds\n", "mama", 69);

        String s = String.format("Yo %s weigh %d pounds", "mama", 69);
        System.out.println(s);
    }

    private static void chars() {
        System.out.println("Here are the printable ascii characters:");
        for (char c = 0x30; c <= 0x7e; c++) {
            System.out.print(c);
        }
        System.out.print("\n\n");

        System.out.println("Here are some unicode waffles:");
        for (char c = 0x10d0; c <= 0x10fa; c++) {
            System.out.print(c);
        }
        System.out.print("\n\n");
    }

   /* Apparently the path of the "linker" or whatever includes the 
    * current directory, so we can use methods from other classes
    * without having to explicitly import them. */

    public void calling_other_classes() {
        Point originOne     = new Point(25, 75);
        Rectangle rectOne   = new Rectangle(originOne, 10, 20);
        Rectangle rectTwo   = new Rectangle(50, 60);

        // Moving the rectangle's origin
        System.out.println("rectOne's " + rectOne.sayOrigin());
        rectOne.move(30, 40);
        System.out.println("rectOne's " + rectOne.sayOrigin());
        System.out.println();

        // Changing the rectangle's width, and observing what changes
        System.out.println("Width  of rectOne: "  + rectOne.width);
        System.out.println("Height of rectOne: "  + rectOne.height);
        System.out.println("Area of rectOne: "    + rectOne.getArea());
        rectOne.width = 20;
        System.out.println("Width  of rectOne: "  + rectOne.width);
        System.out.println("Height of rectOne: "  + rectOne.height);
        System.out.println("Area of rectOne: "    + rectOne.getArea());
        System.out.println();

    }

    class Point {

        int x = 0;
        int y = 0;

        /* A constructor */
        public Point(int a, int b) {
            x = a;
            y = b;
        }
    }

    class Rectangle {

        int width = 0;
        int height = 0;
        Point origin;

        // four constructors
        public Rectangle() {
        	origin = new Point(0, 0);
        }

        public Rectangle(Point p) {
        	origin = p;
        }

        public Rectangle(int w, int h) {
	        origin = new Point(0, 0);
	        width  = w;
	        height = h;
        }

        public Rectangle(Point p, int w, int h) {
	        origin = p;
	        width  = w;
	        height = h;
        }

        /* a method for moving the rectangle */
        public void move(int x, int y) {
            System.out.println("Moving rectangle's origin...");
	        origin.x = x;
	        origin.y = y;
        }

        /* a method for computing the area of the rectangle */
        public int getArea() {
	        return width * height;
        }

        public String sayOrigin() {
            return "Origin: (" + this.origin.x +", "+ this.origin.y +")";
        }
    }
}
