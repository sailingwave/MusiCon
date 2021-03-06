$(document).ready(function() {
    $('#video').hide();

    var source = null;
    var n_clips = 0;
    var video_len = 0;
    var progress_bar = $('#progress_bar');
    var progress_val = $('#progress_val');


    var seconds_to_time = function(sec){
        var hour = Math.floor(sec/60/60);
	    var min  = Math.floor(sec/60%60);
	    var second = Math.floor(sec%60);
	    return (hour < 10 ? "0"+ hour : hour)+ ":"+ (min < 10 ? "0" + min : min) + ":" + (second < 10 ? "0" + second : second);
    }

    $('#gobtn').click(function() {
        $('#input_control').hide();
        $('#video').show();
        $('#goback').hide();

        source = new EventSource('/output/'+encodeURIComponent($("#video_url").val()));

        source.addEventListener('start', event_start);
        source.addEventListener('processing', event_progress);
        source.addEventListener('video', event_video);
    	source.addEventListener('end', event_end);

    	source.onerror = function(e) {
    		console.log(e);
    		source.close();
        };

    });

    var event_start = function(event){
        var data = JSON.parse(event.data);
        var video_title = data.video_title;
        video_len = data.video_len;
        $('#video_title').html("<p>Video title:</p> <h4>"+video_title+"</h4>");
    }

    var event_progress = function(event){
        var data = JSON.parse(event.data);

        //progress bar
        var seconds = data.progress;
        var time = seconds_to_time(seconds);
        progress_val.text(time);
        progress_bar.attr('style','width: '+seconds/video_len*100+'%');

        //Chart of probs
        var is_music_prob = data.is_music_prob;
    }

    var event_video = function(event){
        var data = JSON.parse(event.data);
        var video_url = data.video_url;
        console.log(video_url);

        n_clips = n_clips+1;
        $('#clips').append(`
            <div class="panel panel-default">
                <div class="panel-heading">
                    <h2 class="panel-title">`+ n_clips +`.</h2>
                </div>
                <div class="panel-body">
                    <div style="text-align: center">
                        <iframe width="360" height="270" src="`+video_url+`" frameborder="0" allowfullscreen></iframe>
                    </div>
                </div>
            </div>
            `);
    }

    var event_end = function(event){
        source.close();
        $('#status>h4').text("Done!");
        progress_bar.attr('style','width: 100%');
        $('#stopbtn').hide();
        $('#goback').show();

        if(n_clips == 0){
            $('#clips').append("<h4>Sorry, no clips found.</h4>");
        }
    }

    $('#stopbtn').click(function(){
        $('#status>h4').text("Processing stopped.");
        source.close();
        $('#stopbtn').hide();
        $('#goback').show();

        if(n_clips == 0){
            $('#clips').append("<h4>Sorry, no clips found.</h4>");
        }
    });

    $('#goback').click(function(){
        $('#stopbtn').show();
        $('#goback').hide();
        $('#video').hide();
        $('#input_control').show();
        progress_val.text("00:00:00");
        progress_bar.attr('style','width: 0%');
        $('#status>h4').text("Processing video...");
        var source = null;
        var n_clips = 0;
        var video_len = 0;
    });
})

