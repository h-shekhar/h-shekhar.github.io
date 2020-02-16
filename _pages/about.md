---
layout: home-page
title: About
permalink: /about/
---
<div class="small-wrapper">
  <div class="about-container">
    <section class="about-header">
      <div class="author-image-container">
        <img src="{{site.baseurl}}/assets/img/{% if site.author-pic %}{{site.author-pic}}{% endif %}" alt="{{site.author}}">
      </div>
      <p class="subtitle">
      <!--  {{site.about-author}} -->


        <b><i>"Life is unpredictable. You will never know as to which road can leads you to which destination." ~ Unknown</i></b>
        <br><br>

        I live in New Delhi, India, where I'm an MTech candidate studying Software Engineering at Delhi Technological University. In a past life, I worked as a Consultant reviewing, analyzing, evaluating business processes, business systems, and user needs to achieve process and change improvements that enhanced operational efficiencies. I completed my Bachelor's Degree with a major in Computer Science. My technical areas of interest include <i>Predictive Analytics, Visualization, Machine Learning, Deep Learning, Natural Language Processing, Software Development, Business Process Reengineering.</i> In this personal blog, you will find collection of my thoughts, notes, codes and resources based on my experience in technology.

      </p>
    </section>
    <section class="about-body">
      <ul class="contact-list">
      {% if site.email %}
        <li class="email"><a href="mailto:{{site.email}}"><i class="fa fa-envelope-o"></i></a></li>
      {% else %}
      <li class="email"><a href="mailto:hshekhar0@gmail.com"><i class="fa fa-envelope-o" aria-hidden="true"></i></a></li>
      {% endif %}


      {% if site.website %}
        <li class="website"><a href="http://{{site.website}}" target="_blank"><i class="fa fa-globe"></i></a></li>
      {% else %}
        <li class="website"><a href="https://hshekhar.in" target="_blank"><i class="fa fa-globe" aria-hidden="true"></i></a></li>
      {% endif %}

      {% if site.linkedin %}
        <li class="linkedin"><a href="https://in.linkedin.com/in/{{site.linkedin}}" target="_blank"><i class="fa fa-linkedin"></i></a></li>
      {% else %}
        <li class="linkedin"><a href="https://in.linkedin.com/" target="_blank"><i class="fa fa-linkedin" aria-hidden="true"></i></a></li>
      {% endif %}

      {% if site.facebook %}
        <li class="facebook"><a href="https://facebook.com/{{site.facebook}}" target="_blank"><i class="fa fa-facebook"></i></a></li>
      {% else %}
        <li class="facebook"><a href="https://facebook.com/" target="_blank"><i class="fa fa-facebook" aria-hidden="true"></i></a></li>
      {% endif %}

      {% if site.twitter %}
        <li class="twitter"><a href="https://twitter.com/{{site.twitter}}" target="_blank"><i class="fa fa-twitter"></i></a></li>
      {% else %}
        <li class="twitter"><a href="https://twitter.com/" target="_blank"><i class="fa fa-twitter" aria-hidden="true"></i></a></li>
      {% endif %}

      {% if site.instagram %}
        <li class="instagram"><a href="https://instagram.com/{{site.instagram}}" target="_blank"><i class="fa fa-instagram"></i></a></li>
      {% else %}
        <li class="instagram"><a href="https://instagram.com/" target="_blank"><i class="fa fa-instagram" aria-hidden="true"></i></a></li>
      {% endif %}

      {% if site.medium %}
        <li class="medium"><a href="http://medium.com/{{site.medium}}" target="_blank"><i class="fa fa-medium"></i></a></li>
      {% else %}
        <li class="medium"><a href="http://medium.com/" target="_blank"><i class="fa fa-medium" aria-hidden="true"></i></a></li>
      {% endif %}

      {% if site.github %}
        <li class="github"><a href="http://github.com/{{site.github}}" target="_blank"><i class="fa fa-github"></i></a></li>
      {% else %}
        <li class="github"><a href="http://github.com/" target="_blank"><i class="fa fa-github" aria-hidden="true"></i></a></li>
      {% endif %}

      </ul>
    </section> <!-- End About Body-->
  </div> <!-- End About Container -->
</div> <!-- End Small Wrapper -->
