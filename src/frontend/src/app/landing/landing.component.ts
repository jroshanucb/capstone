import { Component, OnInit } from '@angular/core';
import { faCoffee } from '@fortawesome/free-solid-svg-icons';
import { ApiserviceService } from '../apiservice.service';

declare const loadselectors: any;

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
  isChecked = false;
  setactive = false;

  imageSrc1 = "";
  imageSrc2 = "";
  imageSrc3 = "";
  
  constructor(private service: ApiserviceService) { }
  
  datamapping(data): void {
    this.imageSrc = data.images[0];


    this.imageSrc1 = data.images[0];
    this.imageSrc2 = data.images[1];
    this.imageSrc3 = data.images[2];
    if (this.isChecked) {
      this.imageSrc1 = data.images[3];
      this.imageSrc2 = data.images[4];
      this.imageSrc3 = data.images[5];
    }

    this.animalcount = data.animalcount;
    this.animaltype = data.animaltype
    this.imagegroupid = data.imagegroupid;
    this.dataset=data;
    this.event_id=data.event_id
    console.log(data)
  }


  ngOnInit(): void {
    loadselectors()
    this.service.getImages(this.event_id).subscribe((data:any)=>{
      this.datamapping(data)
    })
  }
  
  radioChange(event) {
    this.imageSrc = event;
  }
  onSpeciesClick(val: any){
    this.animaltype = val;
  }
  onChange() {
    console.log(this.isChecked);
    if (this.isChecked) {
      this.imageSrc1 = this.dataset.images[3];
      this.imageSrc2 = this.dataset.images[4];
      this.imageSrc3 = this.dataset.images[5];
    } else {
      this.imageSrc1 = this.dataset.images[0];
      this.imageSrc2 = this.dataset.images[1];
      this.imageSrc3 = this.dataset.images[2];
    }

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
