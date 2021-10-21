import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Injectable } from '@angular/core';


@Injectable({
  providedIn: 'root'
})
export class ApiserviceService {
  
  constructor(private http: HttpClient) { }

  getImages(){
    return this.http.get("http://localhost:5000/api/v1/resources/newclassify")
  }
  
  postAnnotation(data:string) {
    const headers = { 'content-type': 'application/json'}  
    const body=JSON.stringify(data);
    console.log(body)
    return this.http.post('http://localhost:5000/api/v1/resources/annotate', body,{'headers':headers})
  }

}
