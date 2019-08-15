
const util = {}

util.request = {
    post: function (url, data, callback) {
        oAjax = new XMLHttpRequest();
        oAjax.open('post', url, true);
        oAjax.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        oAjax.send(JSON.stringify(data));
        oAjax.onreadystatechange = function () {
            if (oAjax.readyState == 4) {
                callback(JSON.parse(oAjax.responseText));
            };
        };
    }
}

util.convertCanvasToImage = function(canvas) {  
    var image = new Image();  
    image.src = canvas.toDataURL("image/jpg"); 
    return image;  
} 

util.convertCanvasToBase64 = function(canvas) {  
    return util.convertCanvasToImage(canvas).src.split("base64,")[1];  
} 
