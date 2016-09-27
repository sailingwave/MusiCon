$(document).ready(function() {
    $('#video').hide();

    var source = null;
    var n_clips = 0;
    var video_len = 0;

    $('#gobtn').click(function() {
        $('#input_control').hide();
        $('#video').show();

        source = new EventSource('/output/'+encodeURIComponent($("#video_url").val()));

        source.addEventListener('start', event_start);
        //source.addEventListener('progress', event_progress);
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

    var event_video = function(event){
        var data = JSON.parse(event.data);
        var video_url = data.video_url;
        console.log(video_url);

        n_clips = n_clips+1;
        $('#clips').append('<h5>'+n_clips+'. </h5>');
        $('#clips').append('<iframe width="360" height="270" src="'+video_url+'" frameborder="0" allowfullscreen></iframe>');
    }

    var event_end = function(event){
        source.close();
        $('#status>h4').text("Done!");
    }

    $('#stopbtn').click(function(){
        $('#status>h4').text("Processing stopped.");
        source.close();
    });
})

