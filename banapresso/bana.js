fetch('http://localhost:8080/bana/')
.then((response) => {return response.json()})
.then((data) => {
    const ul = document.querySelector('ul')
    let txt = ''
    data.forEach(el => {
        txt += `
        <li>
            <div>
                <div class="img-wrap"><img src="./bana/${el.img}" alt="${el.img}"></div>
                <div class="store-info">
                    <div class="name">${el.name}</div>
                    <div class="addr">${el.addr}</div>
                </div>
            </div>
        </li>
        `
    });
    ul.innerHTML = txt
    

    var mapContainer = document.getElementById('map'), // 지도를 표시할 div 
    mapOption = {
        center: new kakao.maps.LatLng(36.91671, 127.7226), // 지도의 중심좌표
        level: 12 // 지도의 확대 레벨
    };

    // 지도를 생성합니다    
    var map = new kakao.maps.Map(mapContainer, mapOption);

    data.forEach(element => {
        var geocoder = new kakao.maps.services.Geocoder();
        geocoder.addressSearch(element.addr, function (result, status) {

            // 정상적으로 검색이 완료됐으면 
            if (status === kakao.maps.services.Status.OK) {

                var coords = new kakao.maps.LatLng(result[0].y, result[0].x);

                // 결과값으로 받은 위치를 마커로 표시합니다
                var marker = new kakao.maps.Marker({
                    map: map,
                    position: coords
                });

                // 인포윈도우로 장소에 대한 설명을 표시합니다
                var infowindow = new kakao.maps.InfoWindow({
                    content: element.addr
                });
            }
        });
    })

})