async function fetchData(){
    const API_URL = 'https://api.odcloud.kr/api/3074462/v1/uddi:f046e6e5-58f2-4716-8b74-be62f1a6c6fc_201910221520?page=1&perPage=38'
    const API_KEY = 'Infuser zkiuMPlxABuReB0fSLEQ1+0/7cHD2GyAXRWOwv6qBSOQidvC1jRR+KcrqY3N8FBrcpiST5Tov5gk3kPOdRW/eA=='
    try{
        const res = await fetch(API_URL, {
            headers: {
                'Authorization': API_KEY
            }
        })

        if (!res.ok) {
            throw new Error('API 호출 실패');
        }

        const arr = await res.json();
        const data = arr.data;

        let Obj = []; // 대분류, 중분류 제외 배열
        let btnValue = ''; // 대분류
        let total = 0; // 총 범죄건수
        
        // 대분류 중분류 삭제
        data.forEach((el, idx) => {
            const newObj = { ...el }; // 새로운 객체를 생성하여 원본 객체를 변경하지 않음
            delete newObj['범죄대분류'];
            delete newObj['범죄중분류'];
            Obj.push(newObj);
        });

        // 총 범죄 건수 total값 구하기
        Obj.forEach((el, idx)=>{
            for(let i=0;i<Object.keys(el).length;i++){
                total += Number(Object.values(el)[i]);
            }
        })

        // 범죄 카테고리 클릭 시 데이터 출력
        $('.main-cate').on('click',function () {
            $(this).addClass('on')
            $('.main-cate').not(this).removeClass('on')
            btnValue = $(this).text()
            dataSet(btnValue)

            $('.sec3').fadeIn();
        })
        // 천 단위로 , 찍기
        allTotal = total.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",")

        // dataTot 에 값 출력
        document.querySelector('.dataTot').innerText = `/ ${allTotal}건`


        // 대분류 별 토탈값
        function dataSet(dataValue = '강력범죄') {  
            let Obj = []; // 대분류, 중분류 제외 배열   
            cateTotal = 0 
            for(value in data){
                if(data[value]['범죄대분류'] == dataValue){
                    const newObj = { ...data[value] };
                    delete newObj['범죄대분류'];
                    delete newObj['범죄중분류'];
                    Obj.push(newObj)  
                }
            }
    
            // 대분류 범죄건수 total값 구하기
            Obj.forEach((el, idx)=>{
                for(let i=0;i<Object.keys(el).length;i++){
                    cateTotal += Number(Object.values(el)[i]);
                }
            })
            // 퍼센트값 구하기
            const percentage = ((cateTotal / total) * 100).toFixed(2);
            document.querySelector('#catePer').innerText = `${percentage}%`

            const perBar = document.querySelector('.sec3 .per span');
            $(perBar).css('width', percentage + '%')
            
            
            // 천 단위로 , 찍기
            cateTotal = cateTotal.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",")
            // dataTot 에 값 출력
            document.querySelector('.cateTot').innerText = `${cateTotal}건`

        }    
    } catch (error) {
        console.error('API 요청 오류:', error);
    }
}
fetchData()
