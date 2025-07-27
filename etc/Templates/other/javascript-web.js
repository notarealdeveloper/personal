(function(root, factory) {

    // Browser globals
    root.MYMODULE = factory.call(root);

}(this, function() {

    "use strict";

    function getRuntime() {
        if (typeof window === "object" && window.window === window) {
            return "browser";
        } else if (typeof global === "object" && global.global === global) {
            return "node";
        } else {
            return "unknown";
        }
    }

    return {
        getRuntime: getRuntime,
    };

}));
