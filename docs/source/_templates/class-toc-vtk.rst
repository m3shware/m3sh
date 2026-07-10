.. raw:: html

   <div class="prename">{{ module }}.</div>
   <div class="empty"></div>

{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}
 
.. autoclass:: {{ objname }}
    
   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes
 
   .. autosummary::
      :toctree:
      :template: attribute.rst
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
 
   {% block methods %}
   {% if methods %}
   .. rubric:: Methods
 
   .. autosummary::
      :toctree: 
      :template: method.rst
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
   
   
