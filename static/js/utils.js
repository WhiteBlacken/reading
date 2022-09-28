//截图
function send_base64_gaze_2_remote(x,y,t) {
    let opts = {
        //scale: scale, // 添加的scale 参数
        //canvas: canvas, //自定义 canvas
        //logging: false, //日志开关，便于查看html2canvas的内部执行流程
        //width: width, //dom 原始宽度
        //height: height,
        useCORS: true // 【重要】开启跨域配置
    };
    html2canvas($('body')[0], opts).then(canvas => {
        //document.body.appendChild(canvas);
        // canvas宽度
        let canvasWidth = canvas.width;
        // canvas高度
        let canvasHeight = canvas.height;
        console.log(canvasHeight, canvasWidth);
        //sleep(2);
        // 调用Canvas2Image插件
        // let img = Canvas2Image.convertToImage(canvas, canvasWidth, canvasHeight);
        // let image_data = $(img).attr('src');

        let url = canvas.toDataURL();

        // // 调用Canvas2Image插件
        // Canvas2Image.saveAsImage(canvas, canvasWidth, canvasHeight, 'png', filename);

        let formdata = new FormData();
        formdata.append("image", url.toString());
        formdata.append("x",x);
        formdata.append("y",y);
        formdata.append("t",t);
        $.ajax({
            type: 'POST',
            url: '/image/',
            data: formdata,
            async: false,
            success: function () {
                console.log("success");
            },
            error: function () {
                console.log("error");
            },
            processData: false,
            contentType: false

        })
    });
}

//全屏
function full_screen() {
    if (document.documentElement.RequestFullScreen) {
        document.documentElement.RequestFullScreen();
    }
    //兼容火狐
    if (document.documentElement.mozRequestFullScreen) {
        document.documentElement.mozRequestFullScreen();
    }
    //兼容谷歌等可以webkitRequestFullScreen也可以webkitRequestFullscreen
    if (document.documentElement.webkitRequestFullScreen) {
        document.documentElement.webkitRequestFullScreen();
    }
    //兼容IE,只能写msRequestFullscreen
    if (document.documentElement.msRequestFullscreen) {
        document.documentElement.msRequestFullscreen();
    }
}

//退出全屏
function esc_full_screen() {
    if (document.exitFullScreen) {
        document.exitFullscreen()
    }
    //兼容火狐
    console.log(document.mozExitFullScreen)
    if (document.mozCancelFullScreen) {
        document.mozCancelFullScreen()
    }
    //兼容谷歌等
    if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen()
    }
    //兼容IE
    if (document.msExitFullscreen) {
        document.msExitFullscreen()
    }
}