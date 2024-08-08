{{ name | escape | underline }}

.. currentmodule:: {{ module }}
 
.. autoclass:: {{ objname }}
   :no-inherited-members:
    
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
      :template: function.rst
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
   
   {% block special %}
   {% if all_methods %}
   .. rubric:: Special methods
 
   .. autosummary::
      :toctree: 
      :template: function.rst
   {% for item in all_methods %}
      {%- if item in ['__iter__',
                      '__len__',
                      '__getitem__',
                      '__contains__',
                      '__index__',
                      '__int__',
                     ] %}
         ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
   
