import { Component, OnInit } from '@angular/core';
import { faCoffee } from '@fortawesome/free-solid-svg-icons';
import { ApiserviceService } from '../apiservice.service';


@Component({
  selector: 'app-landing',
  templateUrl: './landing.component.html',
  styleUrls: ['./landing.component.css']
})
export class LandingComponent implements OnInit {
  dataset:any
  data = "";
  firstradio = true;
  imageSrc = "";
  imagegroupid = ""
  animalcount = 0
  animaltype = "";
  animalcheck = false;
  faCoffee = faCoffee;
  event_id = "";
  
  constructor(private service: ApiserviceService) { }
  
  datamapping(data): void {
    this.imageSrc = data.images[0];
    this.animalcount = data.animalcount;
    this.animaltype = data.animaltype
    this.imagegroupid = data.imagegroupid;
    this.dataset=data;
    this.event_id=data.event_id
    console.log(data)
  }


  ngOnInit(): void {
    this.service.getImages(this.event_id).subscribe((data:any)=>{
      this.datamapping(data)
    })
  }
  
  radioChange(event) {
    this.imageSrc = event;
  }

  submitdata(){
    this.data = "{'imagegroupid' :" + this.imagegroupid + ", 'animaltype' :" + this.animaltype + ", 'animalcount' : " + this.animalcount + ", 'event_id' : " + this.event_id + "}";

    this.service.postAnnotation(this.data).subscribe(data => {
      console.log(data)
      this.service.getImages(this.event_id).subscribe((data:any)=>{
        this.datamapping(data)
      });
    });
  }

  submitskip(){
    this.data = "{'imagegroupid' :" + this.imagegroupid + ", 'animaltype' :" + this.animaltype + ", 'animalcount' : " + this.animalcount + ", 'event_id' : " + this.event_id + "}";
    console.log(this.data)
    this.service.getImages(this.event_id).subscribe((data:any)=>{
      this.datamapping(data)
    });
  }
  
}
