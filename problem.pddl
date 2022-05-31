(define (problem world-1)

    (:domain 
        boxworld    
    )
    
    (:objects
        a b c d e x0 y0 x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6
    )
   
; box model
; box  pos    state        locked-by  contains-key  contains-gold
; 1    (x1,y1)  unlocked                a         
; 2    (x2,y2)  locked       a          b           
; 3    (x3,y3)  locked       b          c           True
;            
; agent model
; initially at x0,y0
; has no keys
 
    (:init
        (not (agent-achieved-gem))
        (agent-at x0  y0)
       
	    (unlocked-box x1 y1)
	    (contains-key a x1 y1)

	    (locked-box x2 y2)
	    (contains-key b x2 y2)
	    (opens-by a x2 y2)

        (locked-box x3 y3)
	    (contains-gem x3 y3)
	    (opens-by b x3 y3)
    )
    
    (:goal
	    (agent-achieved-gem)
    )

    (:metric minimize 
        (total-time)
    )
)
