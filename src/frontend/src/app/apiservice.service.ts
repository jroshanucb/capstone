import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Injectable } from '@angular/core';


@Injectable({
  providedIn: 'root'
})
export class ApiserviceService {
  
  constructor(private http: HttpClient) { }

  getImages(event_id:string){
    if (event_id == "")
      return this.http.get("http://54.69.0.194:5000/api/v1/resources/newclassify?event_id=0")
    else
      return this.http.get("http://54.69.0.194:5000/api/v1/resources/newclassify?event_id="+event_id)
  }

  postAnnotation(data:string) {
    const headers = { 'content-type': 'application/json'}  
    const body=JSON.stringify(data);
    console.log(body)
    return this.http.post('http://54.69.0.194:5000/api/v1/resources/annotate', body,{'headers':headers})
  }

}
