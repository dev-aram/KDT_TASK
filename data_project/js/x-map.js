var map = new kakao.maps.Map(document.getElementById('map'), { // 지도를 표시할 div
    center : new kakao.maps.LatLng(36.2683, 127.6358), // 지도의 중심좌표
    level : 13 // 지도의 확대 레벨
});


var positions = [
    {
        title: '안양역', 
        "lat": 37.402707,
        "lng": 126.922044
    },
    {
        title: '안양역 주위 1', 
        "lat": 37.400707,
        "lng": 126.920044
    },
    {
        title: '안양역 주위 2', 
        "lat": 37.403007,
        "lng": 126.925044
    },
    {
        title: '안양역 주위 3',
        "lat": 37.405707,
        "lng": 126.925044
    }
];

var markers = positions.map(function(position) {  // 마커를 배열 단위로 묶음
    return new kakao.maps.Marker({
        position : new kakao.maps.LatLng(position.lat, position.lng)
    });
});
    
var clusterer = new kakao.maps.MarkerClusterer({
        map: map, // 마커들을 클러스터로 관리하고 표시할 지도 객체 
        averageCenter: true, // 클러스터에 포함된 마커들의 평균 위치를 클러스터 마커 위치로 설정 
        minLevel: 5, // 클러스터 할 최소 지도 레벨 
        markers: markers // 클러스터에 마커 추가
});

// 마커 클러스터러에 클릭이벤트를 등록합니다
// 마커 클러스터러를 생성할 때 disableClickZoom을 true로 설정하지 않은 경우
// 이벤트 헨들러로 cluster 객체가 넘어오지 않을 수도 있습니다
kakao.maps.event.addListener(clusterer, 'clusterclick', function(cluster) {

    // 현재 지도 레벨에서 1레벨 확대한 레벨
    var level = map.getLevel()-1;

    // 지도를 클릭된 클러스터의 마커의 위치를 기준으로 확대합니다
    map.setLevel(level, {anchor: cluster.getCenter()});
});