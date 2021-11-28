import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { LandingComponent } from './landing/landing.component';
import { TeamsComponent } from './teams/teams.component';


const routes: Routes = [
  {path:"", component: LandingComponent},
  {path:"annotate", component: LandingComponent},
  {path:"teams", component: TeamsComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }