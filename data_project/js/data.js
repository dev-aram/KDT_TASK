async function fetchData(){
    const API_URL = 'https://api.odcloud.kr/api/3074462/v1/uddi:f046e6e5-58f2-4716-8b74-be62f1a6c6fc_201910221520?perPage=38'
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
        const data = arr.data
        
        for(i of data){
            if(i['범죄대분류'] == '강력범죄'){
                console.log(i);
            }
        }


    } catch (error) {
        console.error('API 요청 오류:', error);
    }
}
fetchData()
