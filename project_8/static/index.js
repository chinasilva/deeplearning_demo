
/* Event */
function distinguish() {
    let c = document.getElementById("canvas");
    let img = util.convertCanvasToBase64(c)

    util.request.post('start', {
        img: img
    }, function (res) {
        var rNumDiv = document.getElementById("result-num-div");
        var height = rNumDiv.offsetHeight;
        rNumDiv.style.cssText = "line-height:" + height + "px";
        rNumDiv.textContent = res.max_number;
        var map = res.probability;


        var tableDiv = document.getElementById("result-table-div");
        var tab = '<table cellspacing="0" cellpadding="0">'
        tab += "<tr style='background:#EEEDED;border:1px solid #EBEEF5'>" + '<th>识别数字</th><th>置信度</th></tr>'

        for (var key in map) {
            tab += '<tr>' + "<td style='border:1px solid #EBEEF5'>" + key + "</td>" + "<td style='border:1px solid #EBEEF5'>" + map[key] + "</td>" + '</tr>';
        }
        tab += '</table>';
        tableDiv.innerHTML = tab;
    })
}

function mouseLeave() {
    alert(0);
}


document.addEventListener('DOMContentLoaded', function () {
    var c, cHeight, cOffsetLeft, cOffsetTop, cWidth, ctx, drawCircle, rgb, container;

    rgb = [000, 000, 000];

    c = document.getElementById("canvas");

    container = document.getElementById("canvas-div");

    var length = container.offsetHeight > container.offsetWidth ? container.offsetWidth : container.offsetHeight;

    c.width = length;

    c.height = c.width;

    cWidth = c.width;

    cHeight = c.height;

    cOffsetTop = c.offsetTop;

    cOffsetLeft = c.offsetLeft;

    ctx = c.getContext("2d");

    ctx.fillStyle = '#fff';

    ctx.fillRect(0, 0, cWidth, cHeight);

    drawCircle = function (ctx, x, y) {
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, 2 * Math.PI);
        ctx.fillStyle = "rgb(" + rgb[0] + ", " + rgb[1] + ", " + rgb[2] + ")";
        return ctx.fill();
    };

    drawLine = function (ctx, ev, size) {
        var canvasX = ev.clientX - cOffsetLeft;
        var canvasY = ev.clientY - cOffsetTop;
        ctx.lineCap = "round";
        ctx.lineWidth = size;
        ctx.strokeStyle = "rgb(" + rgb[0] + ", " + rgb[1] + ", " + rgb[2] + ")";
        ctx.lineTo(canvasX, canvasY);
        ctx.stroke();
    };

    //添加监听事件
    //鼠标按下
    c.addEventListener('mousedown', function (ev) {
        c.canvasMoveUse = true;
        var canvasX = ev.clientX - cOffsetLeft;
        var canvasY = ev.clientY - cOffsetTop;
        ctx.beginPath(); // 移动的起点
        ctx.moveTo(canvasX, canvasY);
    });

    //鼠标移动
    c.addEventListener('mousemove', function (e) {
        if (c.canvasMoveUse) {
            //return drawCircle(ctx, e.clientX - cOffsetLeft, e.clientY - cOffsetTop);
            //划线
            drawLine(ctx, e, cHeight / 10);
        }
    });

    //双击事件 随机改变颜色
    // c.addEventListener('dblclick', function (ev) {
    //     return rgb = [parseInt(Math.random() * 255), parseInt(Math.random() * 255), parseInt(Math.random() * 255)];
    // });

    //鼠标抬起
    c.addEventListener('mouseup', function (ev) {
        console.log("mouseup");
        c.canvasMoveUse = false;
    });

    //鼠标指针离开元素
    c.addEventListener('mouseleave', function (ev) {
        c.canvasMoveUse = false;
    });

    distinguishBtn = document.getElementById('distinguish');

    img = document.getElementById('img');

    //清除
    clear = document.getElementById('clear');

    clear.addEventListener('click', function (ev) {
        ctx.clearRect(0, 0, cWidth, cHeight);
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, cWidth, cHeight);
        var rNumDiv = document.getElementById("result-num-div");
        rNumDiv.textContent = "";

        var tableDiv = document.getElementById("result-table-div");
        tableDiv.innerHTML = "";
    })
});



