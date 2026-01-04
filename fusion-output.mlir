[DEBUG] Recurrence cycle (length 3):
  %65 = neura.reserve : !neura.data<i64, i1>
  %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %109 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %111 -> %65 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %65 = neura.reserve : !neura.data<i64, i1>
  %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %107 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %110 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %111 -> %65 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 2):
  %62 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %112 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %114 -> %62 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 2):
  %59 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %115 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %117 -> %59 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 2):
  %56 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %118 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %120 -> %56 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 2):
  %53 = neura.reserve : !neura.data<i64, i1>
  %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %121 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %123 -> %53 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %53 = neura.reserve : !neura.data<i64, i1>
  %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %102 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %107 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %122 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %123 -> %53 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 2):
  %50 = neura.reserve : !neura.data<i64, i1>
  %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %126 -> %50 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %50 = neura.reserve : !neura.data<i64, i1>
  %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %105 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %107 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %125 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %126 -> %50 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 2):
  %47 = neura.reserve : !neura.data<i64, i1>
  %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %127 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %129 -> %47 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 2):
  %44 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %130 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %132 -> %44 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 2):
  %41 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %133 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %135 -> %41 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 2):
  %38 = neura.reserve : !neura.data<i64, i1>
  %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %136 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %138 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %32 = neura.reserve : !neura.data<i64, i1>
  %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %48 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %139 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %166 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %174 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %176 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %32 = neura.reserve : !neura.data<i64, i1>
  %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %48 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %139 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %166 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %175 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %176 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %29 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %45 = "neura.data_mov"(%31) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %177 = "neura.data_mov"(%150) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %179 -> %29 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 4):
  %26 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %42 = "neura.data_mov"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %151 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %180 = "neura.data_mov"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %182 -> %26 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 4):
  %23 = neura.reserve : !neura.data<i64, i1>
  %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %39 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %154 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %183 = "neura.data_mov"(%156) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %185 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %23 = neura.reserve : !neura.data<i64, i1>
  %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %66 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %155 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %183 = "neura.data_mov"(%156) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %185 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 9):
  %23 = neura.reserve : !neura.data<i64, i1>
  %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %66 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %140 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %166 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %184 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %185 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 9):
  %23 = neura.reserve : !neura.data<i64, i1>
  %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %66 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %143 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %167 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %184 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %185 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 8):
  %23 = neura.reserve : !neura.data<i64, i1>
  %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %66 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %146 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %170 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %184 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %185 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %20 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %63 = "neura.data_mov"(%22) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %157 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %186 = "neura.data_mov"(%159) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %188 -> %20 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 4):
  %17 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %60 = "neura.data_mov"(%19) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %160 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %189 = "neura.data_mov"(%162) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %191 -> %17 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 4):
  %14 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %57 = "neura.data_mov"(%16) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %163 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %192 = "neura.data_mov"(%165) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %194 -> %14 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[DEBUG] Recurrence cycle (length 4):
  %11 = neura.reserve : !neura.data<i64, i1>
  %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %54 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %142 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %195 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %197 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %11 = neura.reserve : !neura.data<i64, i1>
  %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %54 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %102 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %143 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %195 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %197 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 9):
  %11 = neura.reserve : !neura.data<i64, i1>
  %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %54 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %102 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %140 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %166 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %196 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %197 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %11 = neura.reserve : !neura.data<i64, i1>
  %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %54 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %142 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %167 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %196 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %197 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 9):
  %11 = neura.reserve : !neura.data<i64, i1>
  %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %54 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %102 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %143 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %167 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %196 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %197 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 8):
  %11 = neura.reserve : !neura.data<i64, i1>
  %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %54 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %102 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %146 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %170 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %196 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %197 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 4):
  %8 = neura.reserve : !neura.data<i64, i1>
  %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %51 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %145 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %198 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %200 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 5):
  %8 = neura.reserve : !neura.data<i64, i1>
  %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %51 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %105 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %146 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %198 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %200 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 8):
  %8 = neura.reserve : !neura.data<i64, i1>
  %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %51 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %105 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %140 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %166 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %199 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %200 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 8):
  %8 = neura.reserve : !neura.data<i64, i1>
  %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %51 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %105 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %143 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %167 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %199 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %200 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 6):
  %8 = neura.reserve : !neura.data<i64, i1>
  %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %51 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %145 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %170 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %199 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %200 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
[DEBUG] Recurrence cycle (length 7):
  %8 = neura.reserve : !neura.data<i64, i1>
  %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %51 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %105 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %146 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %170 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %199 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %200 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Longest recurrence cycle (length 9):
%23 = neura.reserve : !neura.data<i64, i1>
%25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
%66 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
%101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
%104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
%140 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
%141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
%166 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
%169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
%171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
%172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
%173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
%184 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
%185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
neura.ctrl_mov %185 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %8 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %11 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %14 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %17 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %20 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %23 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %26 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %29 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %32 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %38 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %41 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %44 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %47 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %50 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %53 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %56 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %59 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %62 = neura.reserve : !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %65 = neura.reserve : !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.yield
[MapToAcceleratorPass] Topologically sorted op: %18 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %21 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %27 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %15 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %30 = "neura.data_mov"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %24 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %33 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %12 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %9 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %60 = "neura.data_mov"(%19) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %63 = "neura.data_mov"(%22) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %42 = "neura.data_mov"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %35 = "neura.data_mov"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %57 = "neura.data_mov"(%16) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %73 = "neura.data_mov"(%31) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %45 = "neura.data_mov"(%31) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %66 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %39 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %74 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %48 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %36 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %54 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %51 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %160 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %115 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %76 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %157 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %112 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %68 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %151 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %133 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %163 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %118 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %92 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %148 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %130 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %93 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %78 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %69 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %154 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %136 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %83 = "neura.data_mov"(%75) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %139 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %127 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %77 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %100 = "neura.data_mov"(%37) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %88 = "neura.data_mov"(%37) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %142 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %121 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %102 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %145 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %124 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %105 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %95 = "neura.data_mov"(%94) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %87 = "neura.data_mov"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %71 = "neura.data_mov"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %90 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %80 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %97 = "neura.data_mov"(%89) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %109 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %84 = "neura.data_mov"(%72) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %96 = "neura.data_mov"(%91) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %82 = "neura.data_mov"(%81) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %164 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %161 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %158 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %155 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %152 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %149 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %146 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %143 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %140 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %107 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %99 = "neura.data_mov"(%98) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %86 = "neura.data_mov"(%85) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
[MapToAcceleratorPass] Topologically sorted op: %192 = "neura.data_mov"(%165) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %189 = "neura.data_mov"(%162) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %186 = "neura.data_mov"(%159) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %183 = "neura.data_mov"(%156) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %180 = "neura.data_mov"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %177 = "neura.data_mov"(%150) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %198 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %170 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %195 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %167 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %166 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %137 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %134 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %131 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %128 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %125 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %122 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %119 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %116 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %113 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %110 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[MapToAcceleratorPass] Topologically sorted op: "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[MapToAcceleratorPass] Topologically sorted op: %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %174 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %138 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %135 -> %41 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %132 -> %44 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %129 -> %47 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %126 -> %50 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %123 -> %53 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %120 -> %56 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %117 -> %59 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %114 -> %62 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %111 -> %65 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %201 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %202 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %204 = "neura.data_mov"(%203) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %199 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %196 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %193 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %190 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %187 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %184 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %181 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %178 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %175 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.return_void %204 : !neura.data<i1, i1>
[MapToAcceleratorPass] Topologically sorted op: %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %200 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %197 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %194 -> %14 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %191 -> %17 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %188 -> %20 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %185 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %182 -> %26 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %179 -> %29 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
[MapToAcceleratorPass] Topologically sorted op: neura.ctrl_mov %176 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 0: 6 ops
  %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
  %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
  %11 = neura.reserve : !neura.data<i64, i1>
  %23 = neura.reserve : !neura.data<i64, i1>
  %24 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %12 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 1: 10 ops
  %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
  %8 = neura.reserve : !neura.data<i64, i1>
  %53 = neura.reserve : !neura.data<i64, i1>
  %65 = neura.reserve : !neura.data<i64, i1>
  %9 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %66 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %39 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %54 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 2: 11 ops
  %32 = neura.reserve : !neura.data<i64, i1>
  %50 = neura.reserve : !neura.data<i64, i1>
  %33 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %51 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %142 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %121 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %102 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 3: 16 ops
  %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
  %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
  %17 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %20 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %47 = neura.reserve : !neura.data<i64, i1>
  %18 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %21 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %48 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %145 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %124 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %105 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %109 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 4: 21 ops
  %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
  %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
  %14 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %26 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %59 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %62 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %27 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %15 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %60 = "neura.data_mov"(%19) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %63 = "neura.data_mov"(%22) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %139 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %127 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %155 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %146 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %143 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %140 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %107 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] ALAP Bucket Level 5: 33 ops
  %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
  %29 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %56 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %30 = "neura.data_mov"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %42 = "neura.data_mov"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %35 = "neura.data_mov"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %57 = "neura.data_mov"(%16) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %36 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %160 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %115 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %76 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %157 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %112 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %68 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %78 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %69 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %77 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %198 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %170 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %195 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %167 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %166 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %125 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %122 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %110 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] ALAP Bucket Level 6: 28 ops
  %38 = neura.reserve : !neura.data<i64, i1>
  %41 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %44 = neura.reserve : !neura.data<!llvm.ptr, i1>
  %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %73 = "neura.data_mov"(%31) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %45 = "neura.data_mov"(%31) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %74 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %163 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %118 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %92 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %93 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %88 = "neura.data_mov"(%37) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
  %71 = "neura.data_mov"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %90 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %80 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
  %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %174 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  neura.ctrl_mov %126 -> %50 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %123 -> %53 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %111 -> %65 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP Bucket Level 7: 30 ops
  %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
  %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
  %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
  %151 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %133 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %148 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %130 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %154 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %136 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %83 = "neura.data_mov"(%75) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
  %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %95 = "neura.data_mov"(%94) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %97 = "neura.data_mov"(%89) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
  %84 = "neura.data_mov"(%72) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %96 = "neura.data_mov"(%91) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %82 = "neura.data_mov"(%81) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %164 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %161 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %158 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %152 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %149 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
  %201 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %202 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] ALAP Bucket Level 8: 37 ops
  %100 = "neura.data_mov"(%37) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %87 = "neura.data_mov"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
  %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %99 = "neura.data_mov"(%98) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %86 = "neura.data_mov"(%85) : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
  %192 = "neura.data_mov"(%165) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %189 = "neura.data_mov"(%162) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %186 = "neura.data_mov"(%159) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %183 = "neura.data_mov"(%156) : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
  %180 = "neura.data_mov"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %177 = "neura.data_mov"(%150) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
  %137 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %134 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %131 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %128 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %119 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %116 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %113 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
  %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %204 = "neura.data_mov"(%203) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %199 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %196 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %193 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %190 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %187 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %184 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %181 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %178 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
  %175 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[MapToAcceleratorPass] ALAP Bucket Level 9: 36 ops
  neura.yield
  "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
  "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
  %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %138 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %135 -> %41 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %132 -> %44 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %129 -> %47 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %120 -> %56 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %117 -> %59 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %114 -> %62 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.return_void %204 : !neura.data<i1, i1>
  %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
  %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
  neura.ctrl_mov %200 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %197 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %194 -> %14 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %191 -> %17 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %188 -> %20 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %185 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1>
  neura.ctrl_mov %182 -> %26 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %179 -> %29 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
  neura.ctrl_mov %176 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1>
[MapToAcceleratorPass] ALAP sorted op: %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %11 = neura.reserve : !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %23 = neura.reserve : !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %24 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %12 = "neura.data_mov"(%6) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1> (ALAP level: 0)
[MapToAcceleratorPass] ALAP sorted op: %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %8 = neura.reserve : !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %53 = neura.reserve : !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %65 = neura.reserve : !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %9 = "neura.data_mov"(%7) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %66 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %39 = "neura.data_mov"(%25) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %54 = "neura.data_mov"(%13) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1> (ALAP level: 1)
[MapToAcceleratorPass] ALAP sorted op: %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %32 = neura.reserve : !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %50 = neura.reserve : !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %33 = "neura.data_mov"(%5) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %51 = "neura.data_mov"(%10) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %101 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %142 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %121 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %102 = "neura.data_mov"(%55) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 2)
[MapToAcceleratorPass] ALAP sorted op: %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %17 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %20 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %47 = neura.reserve : !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %18 = "neura.data_mov"(%0) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %21 = "neura.data_mov"(%1) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %48 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %145 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %124 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %105 = "neura.data_mov"(%52) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %109 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %104 = "neura.data_mov"(%103) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (ALAP level: 3)
[MapToAcceleratorPass] ALAP sorted op: %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %14 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %26 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %59 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %62 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %27 = "neura.data_mov"(%2) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %15 = "neura.data_mov"(%3) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %60 = "neura.data_mov"(%19) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %63 = "neura.data_mov"(%22) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %139 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %127 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %155 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %146 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %143 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %140 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %107 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (ALAP level: 4)
[MapToAcceleratorPass] ALAP sorted op: %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %29 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %56 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %30 = "neura.data_mov"(%4) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %42 = "neura.data_mov"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %35 = "neura.data_mov"(%28) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %57 = "neura.data_mov"(%16) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %36 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %160 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %115 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %76 = "neura.data_mov"(%61) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %157 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %112 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %68 = "neura.data_mov"(%64) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %78 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %69 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %77 = "neura.data_mov"(%49) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %198 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %170 = "neura.data_mov"(%147) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %195 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %167 = "neura.data_mov"(%144) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %166 = "neura.data_mov"(%141) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %125 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %122 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %110 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (ALAP level: 5)
[MapToAcceleratorPass] ALAP sorted op: %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %38 = neura.reserve : !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %41 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %44 = neura.reserve : !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %73 = "neura.data_mov"(%31) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %45 = "neura.data_mov"(%31) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %74 = "neura.data_mov"(%34) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %163 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %118 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %92 = "neura.data_mov"(%58) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %93 = "neura.data_mov"(%67) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %88 = "neura.data_mov"(%37) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %71 = "neura.data_mov"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %90 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %80 = "neura.data_mov"(%79) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %174 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %169 = "neura.data_mov"(%168) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %126 -> %50 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %123 -> %53 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %111 -> %65 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 6)
[MapToAcceleratorPass] ALAP sorted op: %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %151 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %133 = "neura.data_mov"(%43) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %148 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %130 = "neura.data_mov"(%46) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %154 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %136 = "neura.data_mov"(%40) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %83 = "neura.data_mov"(%75) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %95 = "neura.data_mov"(%94) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %97 = "neura.data_mov"(%89) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %84 = "neura.data_mov"(%72) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %96 = "neura.data_mov"(%91) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %82 = "neura.data_mov"(%81) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %164 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %161 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %158 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %152 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %149 = "neura.data_mov"(%106) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %201 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %202 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %172 = "neura.data_mov"(%171) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 7)
[MapToAcceleratorPass] ALAP sorted op: %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %100 = "neura.data_mov"(%37) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %87 = "neura.data_mov"(%70) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %99 = "neura.data_mov"(%98) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %86 = "neura.data_mov"(%85) : (!neura.data<i32, i1>) -> !neura.data<i32, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %192 = "neura.data_mov"(%165) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %189 = "neura.data_mov"(%162) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %186 = "neura.data_mov"(%159) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %183 = "neura.data_mov"(%156) : (!neura.data<i64, i1>) -> !neura.data<i64, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %180 = "neura.data_mov"(%153) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %177 = "neura.data_mov"(%150) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %137 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %134 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %131 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %128 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %119 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %116 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %113 = "neura.data_mov"(%108) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %204 = "neura.data_mov"(%203) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %199 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %196 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %193 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %190 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %187 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %184 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %181 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %178 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %175 = "neura.data_mov"(%173) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (ALAP level: 8)
[MapToAcceleratorPass] ALAP sorted op: %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %138 -> %38 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %135 -> %41 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %132 -> %44 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %129 -> %47 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %120 -> %56 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %117 -> %59 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %114 -> %62 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %200 -> %8 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %197 -> %11 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %194 -> %14 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %191 -> %17 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %188 -> %20 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %185 -> %23 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %182 -> %26 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %179 -> %29 : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.ctrl_mov %176 -> %32 : !neura.data<i64, i1> !neura.data<i64, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.return_void %204 : !neura.data<i1, i1> (ALAP level: 9)
[MapToAcceleratorPass] ALAP sorted op: neura.yield (ALAP level: 9)
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 228 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 152 non-materialized operations, 76 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (level: 0)
1 %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1> (level: 0)
2 %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
3 %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
4 %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1> (level: 1)
5 %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
6 %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
7 %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
8 %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
9 %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
10 %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
12 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
13 %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 4)
14 %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 4)
15 %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
16 %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
17 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
18 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
19 %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 5)
20 %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
21 %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
22 %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
23 %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
24 %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
25 %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
26 %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
27 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
28 %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
29 %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
30 %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
31 %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
32 %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
33 %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 6)
34 %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
35 %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
36 %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
37 %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 7)
38 %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
39 %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
40 %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 7)
41 %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
42 %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
43 %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
44 %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
45 %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
46 %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
47 %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 8)
48 %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
49 %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
50 %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
51 %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
52 %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
53 %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
54 %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
55 %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
56 %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1> (level: 8)
57 %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
58 %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
59 %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
60 %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
61 %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
62 %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
63 %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
64 %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
65 %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
66 %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
67 %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
68 %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
69 %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
70 %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
71 %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
72 %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
73 "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
74 "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
75 neura.return_void %204 : !neura.data<i1, i1> (level: 9)
[HeuristicMapping] Found 144 candidate locations for operation: %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=0
[HeuristicMapping] Successfully mapped operation %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 143 candidate locations for operation: %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#15 @t=0
[HeuristicMapping] Successfully mapped operation %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 125 candidate locations for operation: %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=1
[tryRouteDataMove] Routing from Tile#10 @t=0 to Tile#10 @t=1
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 109 candidate locations for operation: %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=1
[tryRouteDataMove] Routing from Tile#15 @t=0 to Tile#14 @t=1
[HeuristicMapping] Successfully mapped operation %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 140 candidate locations for operation: %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=1
[HeuristicMapping] Successfully mapped operation %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 122 candidate locations for operation: %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=2
[tryRouteDataMove] Routing from Tile#10 @t=1 to Tile#10 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 113 candidate locations for operation: %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=2
[tryRouteDataMove] Routing from Tile#14 @t=1 to Tile#14 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #448
[HeuristicMapping] Successfully mapped operation %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 121 candidate locations for operation: %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=2
[tryRouteDataMove] Routing from Tile#9 @t=1 to Tile#9 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #288
[HeuristicMapping] Successfully mapped operation %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 135 candidate locations for operation: %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=3
[tryRouteDataMove] Routing from Tile#10 @t=0 to Tile#10 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #321
[HeuristicMapping] Successfully mapped operation %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 118 candidate locations for operation: %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=3
[tryRouteDataMove] Routing from Tile#9 @t=2 to Tile#9 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #288
[HeuristicMapping] Successfully mapped operation %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 107 candidate locations for operation: %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=3
[tryRouteDataMove] Routing from Tile#10 @t=2 to Tile#14 @t=3
[tryRouteDataMove] Routing from Tile#14 @t=2 to Tile#14 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #448
[HeuristicMapping] Successfully mapped operation %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 133 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#3 @t=3
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 132 candidate locations for operation: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=3
[HeuristicMapping] Successfully mapped operation %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 103 candidate locations for operation: %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=4
[tryRouteDataMove] Routing from Tile#14 @t=3 to Tile#10 @t=4
[tryRouteDataMove] Routing from Tile#9 @t=3 to Tile#10 @t=4
[HeuristicMapping] Successfully mapped operation %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 113 candidate locations for operation: %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=4
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#9 @t=4
[HeuristicMapping] Successfully mapped operation %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 98 candidate locations for operation: %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=4
[tryRouteDataMove] Routing from Tile#3 @t=3 to Tile#7 @t=4
[HeuristicMapping] Successfully mapped operation %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 105 candidate locations for operation: %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=4
[tryRouteDataMove] Routing from Tile#8 @t=3 to Tile#8 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #256
[HeuristicMapping] Successfully mapped operation %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 127 candidate locations for operation: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=4
[HeuristicMapping] Successfully mapped operation %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 126 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=4
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 108 candidate locations for operation: %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=5
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#10 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 100 candidate locations for operation: %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=5
[tryRouteDataMove] Routing from Tile#7 @t=4 to Tile#6 @t=5
[HeuristicMapping] Successfully mapped operation %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 100 candidate locations for operation: %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=5
[tryRouteDataMove] Routing from Tile#8 @t=4 to Tile#9 @t=5
[HeuristicMapping] Successfully mapped operation %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 106 candidate locations for operation: %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=5
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#5 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 104 candidate locations for operation: %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=6
[tryRouteDataMove] Routing from Tile#9 @t=3 to Tile#10 @t=6
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#10 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #321
[HeuristicMapping] Successfully mapped operation %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 104 candidate locations for operation: %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=5
[tryRouteDataMove] Routing from Tile#14 @t=2 to Tile#14 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #449
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#14 @t=5
[HeuristicMapping] Successfully mapped operation %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 104 candidate locations for operation: %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=6
[tryRouteDataMove] Routing from Tile#6 @t=4 to Tile#6 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #192
[HeuristicMapping] Successfully mapped operation %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 96 candidate locations for operation: %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=6
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#9 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #288
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#9 @t=6
[HeuristicMapping] Successfully mapped operation %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 117 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=5
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 99 candidate locations for operation: %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#6 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #192
[HeuristicMapping] Successfully mapped operation %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 96 candidate locations for operation: %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#10 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#10 @t=7
[tryRouteDataMove] Routing from Tile#10 @t=2 to Tile#10 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #322
[HeuristicMapping] Successfully mapped operation %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 91 candidate locations for operation: %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=6
[tryRouteDataMove] Routing from Tile#11 @t=5 to Tile#11 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #352
[HeuristicMapping] Successfully mapped operation %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 98 candidate locations for operation: %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#5 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#5 @t=6
[HeuristicMapping] Successfully mapped operation %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 96 candidate locations for operation: %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#9 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #290
[tryRouteDataMove] Routing from Tile#10 @t=2 to Tile#9 @t=7
[HeuristicMapping] Successfully mapped operation %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 90 candidate locations for operation: %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#13 @t=7
[tryRouteDataMove] Routing from Tile#14 @t=5 to Tile#13 @t=7
[HeuristicMapping] Successfully mapped operation %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 46 candidate locations for operation: %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=3 to Tile#9 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #292
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#9 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=8 to Tile#9 @t=12
[tryRouteDataMove] Successfully routed on same tile using Register #289
[HeuristicMapping] Successfully mapped operation %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 26 candidate locations for operation: %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=6
[tryRouteDataMove] Routing from Tile#14 @t=2 to Tile#14 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #450
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#14 @t=6
[tryRouteDataMove] Routing from Tile#14 @t=6 to Tile#14 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #449
[HeuristicMapping] Successfully mapped operation %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 30 candidate locations for operation: %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=8
[tryRouteDataMove] Routing from Tile#14 @t=3 to Tile#10 @t=8
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#10 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #324
[tryRouteDataMove] Routing from Tile#10 @t=8 to Tile#10 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #322
[HeuristicMapping] Successfully mapped operation %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 82 candidate locations for operation: %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=8
[tryRouteDataMove] Routing from Tile#13 @t=7 to Tile#13 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #416
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#13 @t=8
[HeuristicMapping] Successfully mapped operation %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 100 candidate locations for operation: %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#5 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #162
[HeuristicMapping] Successfully mapped operation %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 85 candidate locations for operation: %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=7
[tryRouteDataMove] Routing from Tile#11 @t=6 to Tile#11 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #352
[HeuristicMapping] Successfully mapped operation %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 104 candidate locations for operation: %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=7
[tryRouteDataMove] Routing from Tile#10 @t=1 to Tile#14 @t=7
[HeuristicMapping] Successfully mapped operation %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 84 candidate locations for operation: %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=8
[tryRouteDataMove] Routing from Tile#11 @t=6 to Tile#11 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #353
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#11 @t=8
[HeuristicMapping] Successfully mapped operation %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 89 candidate locations for operation: %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#6 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #192
[tryRouteDataMove] Routing from Tile#10 @t=2 to Tile#6 @t=8
[HeuristicMapping] Successfully mapped operation %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 90 candidate locations for operation: %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#4 @t=7
[HeuristicMapping] Successfully mapped operation %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 86 candidate locations for operation: %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=7 to Tile#5 @t=8
[HeuristicMapping] Successfully mapped operation %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 84 candidate locations for operation: %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=8
[tryRouteDataMove] Routing from Tile#10 @t=7 to Tile#14 @t=8
[HeuristicMapping] Successfully mapped operation %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 83 candidate locations for operation: %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=9
[tryRouteDataMove] Routing from Tile#10 @t=7 to Tile#11 @t=9
[HeuristicMapping] Successfully mapped operation %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 76 candidate locations for operation: %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=9
[tryRouteDataMove] Routing from Tile#13 @t=8 to Tile#9 @t=9
[HeuristicMapping] Successfully mapped operation %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 65 candidate locations for operation: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=10
[tryRouteDataMove] Routing from Tile#6 @t=8 to Tile#6 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #192
[tryRouteDataMove] Routing from Tile#14 @t=8 to Tile#6 @t=10
[tryRouteDataMove] Routing from Tile#4 @t=7 to Tile#6 @t=10
[HeuristicMapping] Successfully mapped operation %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 69 candidate locations for operation: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=11
[tryRouteDataMove] Routing from Tile#11 @t=9 to Tile#11 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #352
[tryRouteDataMove] Routing from Tile#11 @t=8 to Tile#11 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #353
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#11 @t=11
[HeuristicMapping] Successfully mapped operation %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 82 candidate locations for operation: %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#6 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #194
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#6 @t=9
[HeuristicMapping] Successfully mapped operation %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 90 candidate locations for operation: %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#7 @t=8
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#7 @t=8
[HeuristicMapping] Successfully mapped operation %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 88 candidate locations for operation: %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=9
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#5 @t=9
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#5 @t=9
[HeuristicMapping] Successfully mapped operation %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 72 candidate locations for operation: %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=9
[tryRouteDataMove] Routing from Tile#14 @t=7 to Tile#14 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #448
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#14 @t=9
[HeuristicMapping] Successfully mapped operation %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 78 candidate locations for operation: %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=10
[tryRouteDataMove] Routing from Tile#5 @t=7 to Tile#5 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #162
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#5 @t=10
[HeuristicMapping] Successfully mapped operation %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 70 candidate locations for operation: %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#15 @t=8
[tryRouteDataMove] Routing from Tile#11 @t=7 to Tile#15 @t=8
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#15 @t=8
[HeuristicMapping] Successfully mapped operation %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 69 candidate locations for operation: %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=9
[tryRouteDataMove] Routing from Tile#13 @t=8 to Tile#13 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #416
[tryRouteDataMove] Routing from Tile#13 @t=8 to Tile#13 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #417
[HeuristicMapping] Successfully mapped operation %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[HeuristicMapping] Found 37 candidate locations for operation: %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=10
[tryRouteDataMove] Routing from Tile#14 @t=7 to Tile#13 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#13 @t=10
[tryRouteDataMove] Routing from Tile#13 @t=10 to Tile#14 @t=16
[HeuristicMapping] Successfully mapped operation %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 47 candidate locations for operation: %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=11
[tryRouteDataMove] Routing from Tile#5 @t=7 to Tile#5 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #164
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#5 @t=11
[tryRouteDataMove] Routing from Tile#5 @t=11 to Tile#5 @t=16
[tryRouteDataMove] Successfully routed on same tile using Register #163
[HeuristicMapping] Successfully mapped operation %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 38 candidate locations for operation: %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=10
[tryRouteDataMove] Routing from Tile#11 @t=7 to Tile#11 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #355
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#11 @t=10
[tryRouteDataMove] Routing from Tile#11 @t=10 to Tile#11 @t=16
[tryRouteDataMove] Successfully routed on same tile using Register #355
[HeuristicMapping] Successfully mapped operation %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 12 candidate locations for operation: %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=10
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#8 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#8 @t=10
[tryRouteDataMove] Routing from Tile#8 @t=10 to Tile#9 @t=13
[HeuristicMapping] Successfully mapped operation %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 37 candidate locations for operation: %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=11
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#6 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #196
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#6 @t=11
[tryRouteDataMove] Routing from Tile#6 @t=11 to Tile#6 @t=16
[tryRouteDataMove] Successfully routed on same tile using Register #194
[HeuristicMapping] Successfully mapped operation %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 13 candidate locations for operation: %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=12
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#6 @t=12
[tryRouteDataMove] Successfully routed on same tile using Register #197
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#6 @t=12
[tryRouteDataMove] Routing from Tile#6 @t=12 to Tile#6 @t=14
[tryRouteDataMove] Successfully routed on same tile using Register #195
[HeuristicMapping] Successfully mapped operation %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 12 candidate locations for operation: %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=11
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#8 @t=11
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#8 @t=11
[tryRouteDataMove] Routing from Tile#8 @t=11 to Tile#9 @t=14
[HeuristicMapping] Successfully mapped operation %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] No candidate locations found for operation: %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Backtracking to operation 63(depth = 1)
[HeuristicMapping] Found 12 candidate locations for operation: %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] All 1 locations for 63 tried, backtracking...
[HeuristicMapping] Backtracking to operation 62 (depth = 2).
[HeuristicMapping] Max backtrack depth exceeded: 2 > 1.
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 228 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 152 non-materialized operations, 76 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (level: 0)
1 %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1> (level: 0)
2 %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
3 %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
4 %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1> (level: 1)
5 %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
6 %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
7 %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
8 %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
9 %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
10 %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
12 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
13 %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 4)
14 %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 4)
15 %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
16 %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
17 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
18 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
19 %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 5)
20 %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
21 %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
22 %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
23 %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
24 %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
25 %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
26 %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
27 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
28 %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
29 %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
30 %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
31 %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
32 %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
33 %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 6)
34 %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
35 %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
36 %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
37 %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 7)
38 %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
39 %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
40 %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 7)
41 %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
42 %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
43 %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
44 %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
45 %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
46 %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
47 %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 8)
48 %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
49 %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
50 %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
51 %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
52 %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
53 %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
54 %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
55 %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
56 %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1> (level: 8)
57 %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
58 %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
59 %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
60 %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
61 %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
62 %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
63 %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
64 %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
65 %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
66 %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
67 %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
68 %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
69 %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
70 %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
71 %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
72 %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
73 "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
74 "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
75 neura.return_void %204 : !neura.data<i1, i1> (level: 9)
[HeuristicMapping] Found 160 candidate locations for operation: %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=0
[HeuristicMapping] Successfully mapped operation %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 159 candidate locations for operation: %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=0
[HeuristicMapping] Successfully mapped operation %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 125 candidate locations for operation: %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=1
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#4 @t=1
[HeuristicMapping] Successfully mapped operation %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 140 candidate locations for operation: %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=1
[tryRouteDataMove] Routing from Tile#5 @t=0 to Tile#5 @t=1
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 156 candidate locations for operation: %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=1
[HeuristicMapping] Successfully mapped operation %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 130 candidate locations for operation: %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=2
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#5 @t=2
[HeuristicMapping] Successfully mapped operation %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 137 candidate locations for operation: %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=2
[tryRouteDataMove] Routing from Tile#5 @t=1 to Tile#6 @t=2
[HeuristicMapping] Successfully mapped operation %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 137 candidate locations for operation: %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=2
[tryRouteDataMove] Routing from Tile#10 @t=1 to Tile#10 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 141 candidate locations for operation: %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=3
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#0 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #0
[HeuristicMapping] Successfully mapped operation %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 135 candidate locations for operation: %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=3
[tryRouteDataMove] Routing from Tile#10 @t=2 to Tile#10 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 128 candidate locations for operation: %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=3
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#6 @t=3
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#6 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #192
[HeuristicMapping] Successfully mapped operation %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 149 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=3
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 148 candidate locations for operation: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=3
[HeuristicMapping] Successfully mapped operation %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 123 candidate locations for operation: %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=4
[tryRouteDataMove] Routing from Tile#6 @t=3 to Tile#10 @t=4
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#10 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 114 candidate locations for operation: %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=4
[tryRouteDataMove] Routing from Tile#0 @t=3 to Tile#4 @t=4
[HeuristicMapping] Successfully mapped operation %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 121 candidate locations for operation: %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=4
[tryRouteDataMove] Routing from Tile#11 @t=3 to Tile#11 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #352
[HeuristicMapping] Successfully mapped operation %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 121 candidate locations for operation: %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=4
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#5 @t=4
[HeuristicMapping] Successfully mapped operation %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 143 candidate locations for operation: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=4
[HeuristicMapping] Successfully mapped operation %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 142 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=4
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 124 candidate locations for operation: %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=5
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#10 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 115 candidate locations for operation: %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=5
[tryRouteDataMove] Routing from Tile#11 @t=4 to Tile#11 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #352
[HeuristicMapping] Successfully mapped operation %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 124 candidate locations for operation: %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=5
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#5 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 122 candidate locations for operation: %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=5
[tryRouteDataMove] Routing from Tile#6 @t=4 to Tile#6 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #192
[HeuristicMapping] Successfully mapped operation %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 121 candidate locations for operation: %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=6
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#10 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #321
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#10 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #322
[HeuristicMapping] Successfully mapped operation %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 120 candidate locations for operation: %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=6
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#6 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #193
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#6 @t=6
[HeuristicMapping] Successfully mapped operation %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 120 candidate locations for operation: %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=5
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#9 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #288
[HeuristicMapping] Successfully mapped operation %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 106 candidate locations for operation: %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=4 to Tile#9 @t=6
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#9 @t=6
[HeuristicMapping] Successfully mapped operation %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 133 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=5
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 116 candidate locations for operation: %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=6
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#5 @t=6
[HeuristicMapping] Successfully mapped operation %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 100 candidate locations for operation: %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=7
[tryRouteDataMove] Routing from Tile#11 @t=5 to Tile#6 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=4 to Tile#6 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#6 @t=7
[HeuristicMapping] Successfully mapped operation %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 102 candidate locations for operation: %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=6
[tryRouteDataMove] Routing from Tile#0 @t=5 to Tile#4 @t=6
[HeuristicMapping] Successfully mapped operation %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 110 candidate locations for operation: %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=6
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#2 @t=6
[tryRouteDataMove] Routing from Tile#0 @t=3 to Tile#2 @t=6
[HeuristicMapping] Successfully mapped operation %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 112 candidate locations for operation: %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#5 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #161
[HeuristicMapping] Successfully mapped operation %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 104 candidate locations for operation: %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#10 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#10 @t=7
[HeuristicMapping] Successfully mapped operation %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 67 candidate locations for operation: %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=8
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#10 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #323
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#10 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #320
[tryRouteDataMove] Routing from Tile#10 @t=8 to Tile#10 @t=13
[tryRouteDataMove] Successfully routed on same tile using Register #321
[HeuristicMapping] Successfully mapped operation %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 51 candidate locations for operation: %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=2 to Tile#6 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #195
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#6 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=8 to Tile#6 @t=12
[tryRouteDataMove] Successfully routed on same tile using Register #192
[HeuristicMapping] Successfully mapped operation %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 48 candidate locations for operation: %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=3 to Tile#9 @t=7
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#9 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=7 to Tile#5 @t=12
[HeuristicMapping] Successfully mapped operation %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 107 candidate locations for operation: %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=9
[tryRouteDataMove] Routing from Tile#10 @t=7 to Tile#10 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #322
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#10 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #324
[HeuristicMapping] Successfully mapped operation %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 116 candidate locations for operation: %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#7 @t=7
[HeuristicMapping] Successfully mapped operation %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 101 candidate locations for operation: %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#4 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #128
[HeuristicMapping] Successfully mapped operation %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 120 candidate locations for operation: %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#8 @t=7
[HeuristicMapping] Successfully mapped operation %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 99 candidate locations for operation: %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#0 @t=7
[tryRouteDataMove] Routing from Tile#0 @t=3 to Tile#0 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #0
[HeuristicMapping] Successfully mapped operation %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 104 candidate locations for operation: %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#5 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #162
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #163
[HeuristicMapping] Successfully mapped operation %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 99 candidate locations for operation: %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=7
[tryRouteDataMove] Routing from Tile#2 @t=6 to Tile#2 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #64
[HeuristicMapping] Successfully mapped operation %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 100 candidate locations for operation: %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=7 to Tile#4 @t=8
[HeuristicMapping] Successfully mapped operation %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 99 candidate locations for operation: %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#7 @t=8
[HeuristicMapping] Successfully mapped operation %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 98 candidate locations for operation: %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#2 @t=8
[HeuristicMapping] Successfully mapped operation %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 99 candidate locations for operation: %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=9 to Tile#10 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 82 candidate locations for operation: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=9
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#6 @t=9
[tryRouteDataMove] Routing from Tile#7 @t=8 to Tile#6 @t=9
[tryRouteDataMove] Routing from Tile#2 @t=7 to Tile#6 @t=9
[HeuristicMapping] Successfully mapped operation %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 77 candidate locations for operation: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=10
[tryRouteDataMove] Routing from Tile#2 @t=8 to Tile#1 @t=10
[tryRouteDataMove] Routing from Tile#0 @t=7 to Tile#1 @t=10
[tryRouteDataMove] Routing from Tile#4 @t=8 to Tile#1 @t=10
[HeuristicMapping] Successfully mapped operation %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 101 candidate locations for operation: %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#9 @t=8
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#9 @t=8
[HeuristicMapping] Successfully mapped operation %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 104 candidate locations for operation: %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=8
[tryRouteDataMove] Routing from Tile#11 @t=5 to Tile#11 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #352
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#11 @t=8
[HeuristicMapping] Successfully mapped operation %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 101 candidate locations for operation: %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=9
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#5 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #164
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#5 @t=9
[HeuristicMapping] Successfully mapped operation %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 88 candidate locations for operation: %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=8
[tryRouteDataMove] Routing from Tile#8 @t=7 to Tile#8 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #256
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#8 @t=8
[HeuristicMapping] Successfully mapped operation %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 87 candidate locations for operation: %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=9
[tryRouteDataMove] Routing from Tile#7 @t=7 to Tile#7 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #224
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#7 @t=9
[HeuristicMapping] Successfully mapped operation %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 83 candidate locations for operation: %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=9
[tryRouteDataMove] Routing from Tile#4 @t=7 to Tile#9 @t=9
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#9 @t=9
[HeuristicMapping] Successfully mapped operation %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 91 candidate locations for operation: %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=9 to Tile#6 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=9 to Tile#6 @t=10
[tryRouteDataMove] Cannot find routing path from Tile#10 @t=9 to Tile#6 @t=10
[HeuristicMapping] Failed to map operation %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1> to candidate location 1/1
[HeuristicMapping] Found 91 candidate locations for operation: %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[HeuristicMapping] All 1 locations for 56 tried, backtracking...
[HeuristicMapping] Backtracking to operation 55 (depth = 1).
[HeuristicMapping] Found 83 candidate locations for operation: %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] All 1 locations for 55 tried, backtracking...
[HeuristicMapping] Backtracking to operation 54 (depth = 2).
[HeuristicMapping] Max backtrack depth exceeded: 2 > 1.
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 228 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 152 non-materialized operations, 76 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (level: 0)
1 %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1> (level: 0)
2 %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
3 %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
4 %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1> (level: 1)
5 %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
6 %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
7 %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
8 %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
9 %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
10 %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
12 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
13 %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 4)
14 %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 4)
15 %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
16 %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
17 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
18 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
19 %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 5)
20 %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
21 %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
22 %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
23 %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
24 %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
25 %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
26 %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
27 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
28 %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
29 %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
30 %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
31 %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
32 %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
33 %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 6)
34 %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
35 %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
36 %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
37 %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 7)
38 %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
39 %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
40 %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 7)
41 %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
42 %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
43 %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
44 %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
45 %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
46 %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
47 %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 8)
48 %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
49 %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
50 %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
51 %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
52 %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
53 %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
54 %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
55 %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
56 %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1> (level: 8)
57 %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
58 %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
59 %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
60 %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
61 %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
62 %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
63 %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
64 %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
65 %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
66 %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
67 %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
68 %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
69 %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
70 %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
71 %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
72 %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
73 "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
74 "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
75 neura.return_void %204 : !neura.data<i1, i1> (level: 9)
[HeuristicMapping] Found 176 candidate locations for operation: %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=0
[HeuristicMapping] Successfully mapped operation %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 175 candidate locations for operation: %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=0
[HeuristicMapping] Successfully mapped operation %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 141 candidate locations for operation: %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=1
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#0 @t=1
[tryRouteDataMove] Successfully routed on same tile using Register #0
[HeuristicMapping] Successfully mapped operation %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 148 candidate locations for operation: %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=1
[tryRouteDataMove] Routing from Tile#4 @t=0 to Tile#5 @t=1
[HeuristicMapping] Successfully mapped operation %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 172 candidate locations for operation: %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=1
[HeuristicMapping] Successfully mapped operation %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 138 candidate locations for operation: %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=2
[tryRouteDataMove] Routing from Tile#0 @t=1 to Tile#4 @t=2
[HeuristicMapping] Successfully mapped operation %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 153 candidate locations for operation: %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=2
[tryRouteDataMove] Routing from Tile#5 @t=1 to Tile#5 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 144 candidate locations for operation: %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=2
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#8 @t=2
[HeuristicMapping] Successfully mapped operation %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 157 candidate locations for operation: %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=3
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#4 @t=3
[HeuristicMapping] Successfully mapped operation %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 142 candidate locations for operation: %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=3
[tryRouteDataMove] Routing from Tile#8 @t=2 to Tile#8 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #256
[HeuristicMapping] Successfully mapped operation %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 139 candidate locations for operation: %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=3
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#5 @t=3
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 165 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#15 @t=3
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 164 candidate locations for operation: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=3
[HeuristicMapping] Successfully mapped operation %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 135 candidate locations for operation: %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=4
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#9 @t=4
[tryRouteDataMove] Routing from Tile#8 @t=3 to Tile#9 @t=4
[HeuristicMapping] Successfully mapped operation %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 138 candidate locations for operation: %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=4
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#5 @t=4
[HeuristicMapping] Successfully mapped operation %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 130 candidate locations for operation: %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=4
[tryRouteDataMove] Routing from Tile#15 @t=3 to Tile#11 @t=4
[HeuristicMapping] Successfully mapped operation %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 137 candidate locations for operation: %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=4
[tryRouteDataMove] Routing from Tile#7 @t=3 to Tile#7 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #224
[HeuristicMapping] Successfully mapped operation %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 159 candidate locations for operation: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=4
[HeuristicMapping] Successfully mapped operation %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 158 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=4
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 140 candidate locations for operation: %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=5
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#9 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #288
[HeuristicMapping] Successfully mapped operation %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 132 candidate locations for operation: %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=5
[tryRouteDataMove] Routing from Tile#11 @t=4 to Tile#11 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #352
[HeuristicMapping] Successfully mapped operation %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 131 candidate locations for operation: %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=5
[tryRouteDataMove] Routing from Tile#7 @t=4 to Tile#6 @t=5
[HeuristicMapping] Successfully mapped operation %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 139 candidate locations for operation: %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=5
[tryRouteDataMove] Routing from Tile#6 @t=4 to Tile#5 @t=5
[HeuristicMapping] Successfully mapped operation %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 136 candidate locations for operation: %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=5
[tryRouteDataMove] Routing from Tile#8 @t=3 to Tile#8 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #256
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#8 @t=5
[HeuristicMapping] Successfully mapped operation %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 137 candidate locations for operation: %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #161
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#5 @t=6
[HeuristicMapping] Successfully mapped operation %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 132 candidate locations for operation: %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=5
[tryRouteDataMove] Routing from Tile#14 @t=4 to Tile#10 @t=5
[HeuristicMapping] Successfully mapped operation %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 127 candidate locations for operation: %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#9 @t=6
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#9 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #289
[HeuristicMapping] Successfully mapped operation %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 149 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=5
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 132 candidate locations for operation: %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=6
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#10 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 120 candidate locations for operation: %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=7
[tryRouteDataMove] Routing from Tile#11 @t=5 to Tile#6 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#6 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#6 @t=7
[HeuristicMapping] Successfully mapped operation %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 124 candidate locations for operation: %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=6
[tryRouteDataMove] Routing from Tile#13 @t=5 to Tile#13 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #416
[HeuristicMapping] Successfully mapped operation %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 130 candidate locations for operation: %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#4 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#4 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #128
[HeuristicMapping] Successfully mapped operation %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 129 candidate locations for operation: %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=6
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#6 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #193
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#6 @t=6
[HeuristicMapping] Successfully mapped operation %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 120 candidate locations for operation: %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#9 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #288
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#9 @t=7
[HeuristicMapping] Successfully mapped operation %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 73 candidate locations for operation: %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=6
[tryRouteDataMove] Routing from Tile#8 @t=3 to Tile#8 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #257
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#8 @t=6
[tryRouteDataMove] Routing from Tile#8 @t=6 to Tile#8 @t=14
[tryRouteDataMove] Successfully routed on same tile using Register #257
[HeuristicMapping] Successfully mapped operation %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 67 candidate locations for operation: %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #162
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#5 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=7 to Tile#5 @t=13
[tryRouteDataMove] Successfully routed on same tile using Register #161
[HeuristicMapping] Successfully mapped operation %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 53 candidate locations for operation: %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#5 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #163
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#5 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#4 @t=13
[HeuristicMapping] Successfully mapped operation %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 122 candidate locations for operation: %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=7 to Tile#9 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #288
[tryRouteDataMove] Routing from Tile#8 @t=5 to Tile#9 @t=8
[HeuristicMapping] Successfully mapped operation %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 131 candidate locations for operation: %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#4 @t=7
[HeuristicMapping] Successfully mapped operation %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 115 candidate locations for operation: %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=7
[tryRouteDataMove] Routing from Tile#13 @t=6 to Tile#13 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #416
[HeuristicMapping] Successfully mapped operation %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 136 candidate locations for operation: %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=7
[tryRouteDataMove] Routing from Tile#0 @t=1 to Tile#0 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #0
[HeuristicMapping] Successfully mapped operation %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 114 candidate locations for operation: %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#12 @t=7
[tryRouteDataMove] Routing from Tile#13 @t=6 to Tile#12 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#12 @t=7
[HeuristicMapping] Successfully mapped operation %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 116 candidate locations for operation: %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=8
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#6 @t=8
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#6 @t=8
[HeuristicMapping] Successfully mapped operation %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 112 candidate locations for operation: %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#8 @t=7
[HeuristicMapping] Successfully mapped operation %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 121 candidate locations for operation: %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#7 @t=7
[HeuristicMapping] Successfully mapped operation %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 114 candidate locations for operation: %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#7 @t=8
[HeuristicMapping] Successfully mapped operation %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 113 candidate locations for operation: %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#2 @t=8
[HeuristicMapping] Successfully mapped operation %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 112 candidate locations for operation: %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=9
[tryRouteDataMove] Routing from Tile#9 @t=8 to Tile#9 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #288
[HeuristicMapping] Successfully mapped operation %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 96 candidate locations for operation: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=10
[tryRouteDataMove] Routing from Tile#6 @t=8 to Tile#6 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #192
[tryRouteDataMove] Routing from Tile#7 @t=8 to Tile#6 @t=10
[tryRouteDataMove] Routing from Tile#8 @t=7 to Tile#6 @t=10
[HeuristicMapping] Successfully mapped operation %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 90 candidate locations for operation: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=10
[tryRouteDataMove] Routing from Tile#2 @t=8 to Tile#5 @t=10
[tryRouteDataMove] Routing from Tile#12 @t=7 to Tile#5 @t=10
[tryRouteDataMove] Routing from Tile#7 @t=7 to Tile#5 @t=10
[HeuristicMapping] Successfully mapped operation %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 119 candidate locations for operation: %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=8
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#10 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #321
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#10 @t=8
[HeuristicMapping] Successfully mapped operation %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 116 candidate locations for operation: %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=8
[tryRouteDataMove] Routing from Tile#11 @t=5 to Tile#11 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #352
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#11 @t=8
[HeuristicMapping] Successfully mapped operation %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 118 candidate locations for operation: %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#10 @t=9
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#10 @t=9
[HeuristicMapping] Successfully mapped operation %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 96 candidate locations for operation: %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=8
[tryRouteDataMove] Routing from Tile#0 @t=7 to Tile#4 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#4 @t=8
[HeuristicMapping] Successfully mapped operation %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 102 candidate locations for operation: %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=8
[tryRouteDataMove] Routing from Tile#4 @t=7 to Tile#8 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#8 @t=8
[HeuristicMapping] Successfully mapped operation %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 104 candidate locations for operation: %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=8
[tryRouteDataMove] Routing from Tile#13 @t=7 to Tile#13 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #416
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#13 @t=8
[HeuristicMapping] Successfully mapped operation %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 102 candidate locations for operation: %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=10
[tryRouteDataMove] Routing from Tile#9 @t=8 to Tile#9 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #289
[tryRouteDataMove] Routing from Tile#9 @t=8 to Tile#9 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #290
[HeuristicMapping] Successfully mapped operation %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[HeuristicMapping] Found 65 candidate locations for operation: %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=9
[tryRouteDataMove] Routing from Tile#0 @t=7 to Tile#1 @t=9
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#1 @t=9
[tryRouteDataMove] Routing from Tile#1 @t=9 to Tile#0 @t=18
[HeuristicMapping] Successfully mapped operation %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 70 candidate locations for operation: %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=9
[tryRouteDataMove] Routing from Tile#4 @t=7 to Tile#8 @t=9
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#8 @t=9
[tryRouteDataMove] Routing from Tile#8 @t=9 to Tile#4 @t=18
[HeuristicMapping] Successfully mapped operation %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 72 candidate locations for operation: %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=9
[tryRouteDataMove] Routing from Tile#13 @t=7 to Tile#13 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #418
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#13 @t=9
[tryRouteDataMove] Routing from Tile#13 @t=9 to Tile#13 @t=18
[tryRouteDataMove] Successfully routed on same tile using Register #417
[HeuristicMapping] Successfully mapped operation %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 52 candidate locations for operation: %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=11
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#5 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #165
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#5 @t=11
[tryRouteDataMove] Routing from Tile#5 @t=11 to Tile#5 @t=15
[tryRouteDataMove] Successfully routed on same tile using Register #164
[HeuristicMapping] Successfully mapped operation %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 62 candidate locations for operation: %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#10 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #324
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#10 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=10 to Tile#10 @t=17
[tryRouteDataMove] Successfully routed on same tile using Register #321
[HeuristicMapping] Successfully mapped operation %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 43 candidate locations for operation: %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=11
[tryRouteDataMove] Routing from Tile#11 @t=5 to Tile#11 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #354
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#11 @t=11
[tryRouteDataMove] Routing from Tile#11 @t=11 to Tile#11 @t=16
[tryRouteDataMove] Successfully routed on same tile using Register #353
[HeuristicMapping] Successfully mapped operation %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 42 candidate locations for operation: %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=11
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#10 @t=11
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#10 @t=11
[tryRouteDataMove] Routing from Tile#10 @t=11 to Tile#6 @t=16
[HeuristicMapping] Successfully mapped operation %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 9 candidate locations for operation: %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=10
[tryRouteDataMove] Routing from Tile#8 @t=5 to Tile#8 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #256
[tryRouteDataMove] Routing from Tile#9 @t=9 to Tile#8 @t=10
[tryRouteDataMove] Routing from Tile#8 @t=10 to Tile#8 @t=13
[tryRouteDataMove] Successfully routed on same tile using Register #256
[HeuristicMapping] Successfully mapped operation %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 2 candidate locations for operation: %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=11
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#9 @t=11
[tryRouteDataMove] Routing from Tile#9 @t=9 to Tile#9 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #288
[tryRouteDataMove] Routing from Tile#9 @t=11 to Tile#5 @t=12
[HeuristicMapping] Successfully mapped operation %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 32 candidate locations for operation: %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=12
[tryRouteDataMove] Routing from Tile#10 @t=8 to Tile#10 @t=12
[tryRouteDataMove] Successfully routed on same tile using Register #322
[tryRouteDataMove] Routing from Tile#9 @t=9 to Tile#10 @t=12
[tryRouteDataMove] Routing from Tile#10 @t=12 to Tile#10 @t=16
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 18 candidate locations for operation: %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=12
[tryRouteDataMove] Routing from Tile#11 @t=8 to Tile#9 @t=12
[tryRouteDataMove] Routing from Tile#9 @t=9 to Tile#9 @t=12
[tryRouteDataMove] Successfully routed on same tile using Register #293
[tryRouteDataMove] Routing from Tile#9 @t=12 to Tile#11 @t=15
[HeuristicMapping] Successfully mapped operation %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 14 candidate locations for operation: %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=12
[tryRouteDataMove] Routing from Tile#10 @t=9 to Tile#6 @t=12
[tryRouteDataMove] Routing from Tile#9 @t=9 to Tile#6 @t=12
[tryRouteDataMove] Routing from Tile#6 @t=12 to Tile#7 @t=15
[HeuristicMapping] Successfully mapped operation %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] No candidate locations found for operation: %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Backtracking to operation 68(depth = 1)
[HeuristicMapping] Found 14 candidate locations for operation: %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] All 1 locations for 68 tried, backtracking...
[HeuristicMapping] Backtracking to operation 67 (depth = 2).
[HeuristicMapping] Max backtrack depth exceeded: 2 > 1.
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 228 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 152 non-materialized operations, 76 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (level: 0)
1 %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1> (level: 0)
2 %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
3 %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
4 %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1> (level: 1)
5 %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
6 %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
7 %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
8 %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
9 %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
10 %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
12 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
13 %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 4)
14 %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 4)
15 %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
16 %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
17 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
18 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
19 %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 5)
20 %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
21 %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
22 %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
23 %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
24 %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
25 %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
26 %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
27 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
28 %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
29 %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
30 %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
31 %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
32 %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
33 %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 6)
34 %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
35 %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
36 %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
37 %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 7)
38 %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
39 %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
40 %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 7)
41 %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
42 %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
43 %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
44 %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
45 %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
46 %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
47 %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 8)
48 %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
49 %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
50 %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
51 %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
52 %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
53 %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
54 %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
55 %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
56 %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1> (level: 8)
57 %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
58 %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
59 %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
60 %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
61 %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
62 %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
63 %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
64 %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
65 %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
66 %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
67 %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
68 %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
69 %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
70 %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
71 %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
72 %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
73 "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
74 "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
75 neura.return_void %204 : !neura.data<i1, i1> (level: 9)
[HeuristicMapping] Found 192 candidate locations for operation: %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=0
[HeuristicMapping] Successfully mapped operation %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 191 candidate locations for operation: %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=0
[HeuristicMapping] Successfully mapped operation %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 157 candidate locations for operation: %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=1
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#4 @t=1
[HeuristicMapping] Successfully mapped operation %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 164 candidate locations for operation: %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=1
[tryRouteDataMove] Routing from Tile#4 @t=0 to Tile#5 @t=1
[HeuristicMapping] Successfully mapped operation %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 188 candidate locations for operation: %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=1
[HeuristicMapping] Successfully mapped operation %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 162 candidate locations for operation: %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=2
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#4 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #128
[HeuristicMapping] Successfully mapped operation %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 169 candidate locations for operation: %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=2
[tryRouteDataMove] Routing from Tile#5 @t=1 to Tile#5 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 161 candidate locations for operation: %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=2
[tryRouteDataMove] Routing from Tile#1 @t=1 to Tile#1 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #32
[HeuristicMapping] Successfully mapped operation %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 173 candidate locations for operation: %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=3
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#4 @t=3
[HeuristicMapping] Successfully mapped operation %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 159 candidate locations for operation: %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=3
[tryRouteDataMove] Routing from Tile#1 @t=2 to Tile#5 @t=3
[HeuristicMapping] Successfully mapped operation %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 154 candidate locations for operation: %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=4
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#5 @t=4
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #161
[HeuristicMapping] Successfully mapped operation %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 181 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=3
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 180 candidate locations for operation: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=3
[HeuristicMapping] Successfully mapped operation %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 162 candidate locations for operation: %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=5
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#5 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#5 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #162
[HeuristicMapping] Successfully mapped operation %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 153 candidate locations for operation: %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=4
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#4 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #128
[HeuristicMapping] Successfully mapped operation %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 153 candidate locations for operation: %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=4
[tryRouteDataMove] Routing from Tile#8 @t=3 to Tile#8 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #256
[HeuristicMapping] Successfully mapped operation %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 153 candidate locations for operation: %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=4
[tryRouteDataMove] Routing from Tile#1 @t=3 to Tile#1 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #32
[HeuristicMapping] Successfully mapped operation %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 175 candidate locations for operation: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=4
[HeuristicMapping] Successfully mapped operation %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 174 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#15 @t=4
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 156 candidate locations for operation: %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#5 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 148 candidate locations for operation: %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=5
[tryRouteDataMove] Routing from Tile#8 @t=4 to Tile#9 @t=5
[HeuristicMapping] Successfully mapped operation %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 147 candidate locations for operation: %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=5
[tryRouteDataMove] Routing from Tile#1 @t=4 to Tile#1 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #32
[HeuristicMapping] Successfully mapped operation %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 149 candidate locations for operation: %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=5
[tryRouteDataMove] Routing from Tile#11 @t=4 to Tile#10 @t=5
[HeuristicMapping] Successfully mapped operation %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 152 candidate locations for operation: %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#5 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #163
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#5 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #161
[HeuristicMapping] Successfully mapped operation %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 151 candidate locations for operation: %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#9 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#9 @t=6
[HeuristicMapping] Successfully mapped operation %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 141 candidate locations for operation: %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=5
[tryRouteDataMove] Routing from Tile#15 @t=4 to Tile#11 @t=5
[HeuristicMapping] Successfully mapped operation %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 148 candidate locations for operation: %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=4 to Tile#4 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #128
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#4 @t=6
[HeuristicMapping] Successfully mapped operation %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 165 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=5
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 143 candidate locations for operation: %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=6
[tryRouteDataMove] Routing from Tile#11 @t=5 to Tile#10 @t=6
[HeuristicMapping] Successfully mapped operation %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 143 candidate locations for operation: %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=6
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#8 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=4 to Tile#8 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#8 @t=6
[HeuristicMapping] Successfully mapped operation %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 139 candidate locations for operation: %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=7
[tryRouteDataMove] Routing from Tile#8 @t=5 to Tile#9 @t=7
[HeuristicMapping] Successfully mapped operation %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 144 candidate locations for operation: %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=6
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#6 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#6 @t=6
[HeuristicMapping] Successfully mapped operation %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 140 candidate locations for operation: %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=6
[tryRouteDataMove] Routing from Tile#1 @t=5 to Tile#0 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#0 @t=6
[HeuristicMapping] Successfully mapped operation %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 132 candidate locations for operation: %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#8 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#8 @t=7
[HeuristicMapping] Successfully mapped operation %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 83 candidate locations for operation: %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#5 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #164
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#5 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#5 @t=15
[tryRouteDataMove] Successfully routed on same tile using Register #162
[HeuristicMapping] Successfully mapped operation %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 70 candidate locations for operation: %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=9
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #165
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#5 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #166
[tryRouteDataMove] Routing from Tile#5 @t=9 to Tile#5 @t=14
[tryRouteDataMove] Successfully routed on same tile using Register #161
[HeuristicMapping] Successfully mapped operation %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 61 candidate locations for operation: %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#4 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#4 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=7 to Tile#4 @t=14
[tryRouteDataMove] Successfully routed on same tile using Register #129
[HeuristicMapping] Successfully mapped operation %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 128 candidate locations for operation: %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=8
[tryRouteDataMove] Routing from Tile#8 @t=7 to Tile#9 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=7 to Tile#9 @t=8
[HeuristicMapping] Successfully mapped operation %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 148 candidate locations for operation: %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=7
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#10 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 136 candidate locations for operation: %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=7 to Tile#10 @t=8
[HeuristicMapping] Successfully mapped operation %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 152 candidate locations for operation: %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#6 @t=7
[HeuristicMapping] Successfully mapped operation %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 133 candidate locations for operation: %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=7 to Tile#8 @t=8
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#8 @t=8
[HeuristicMapping] Successfully mapped operation %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 133 candidate locations for operation: %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=8
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#6 @t=8
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#6 @t=8
[HeuristicMapping] Successfully mapped operation %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 136 candidate locations for operation: %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#7 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#7 @t=7
[HeuristicMapping] Successfully mapped operation %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 124 candidate locations for operation: %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=7
[tryRouteDataMove] Routing from Tile#0 @t=6 to Tile#0 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #0
[HeuristicMapping] Successfully mapped operation %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 127 candidate locations for operation: %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#12 @t=7
[tryRouteDataMove] Routing from Tile#8 @t=6 to Tile#12 @t=7
[HeuristicMapping] Successfully mapped operation %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 126 candidate locations for operation: %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#12 @t=8
[tryRouteDataMove] Routing from Tile#8 @t=6 to Tile#12 @t=8
[HeuristicMapping] Successfully mapped operation %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 128 candidate locations for operation: %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=9
[tryRouteDataMove] Routing from Tile#9 @t=8 to Tile#9 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #288
[HeuristicMapping] Successfully mapped operation %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 114 candidate locations for operation: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=10
[tryRouteDataMove] Routing from Tile#6 @t=8 to Tile#5 @t=10
[tryRouteDataMove] Routing from Tile#12 @t=7 to Tile#5 @t=10
[tryRouteDataMove] Routing from Tile#7 @t=7 to Tile#5 @t=10
[HeuristicMapping] Successfully mapped operation %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 106 candidate locations for operation: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=9
[tryRouteDataMove] Routing from Tile#12 @t=8 to Tile#8 @t=9
[tryRouteDataMove] Routing from Tile#8 @t=8 to Tile#8 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #256
[tryRouteDataMove] Routing from Tile#0 @t=7 to Tile#8 @t=9
[HeuristicMapping] Successfully mapped operation %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 132 candidate locations for operation: %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=9
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#10 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #321
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#10 @t=9
[HeuristicMapping] Successfully mapped operation %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 134 candidate locations for operation: %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=10
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#9 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #289
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#9 @t=10
[HeuristicMapping] Successfully mapped operation %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 128 candidate locations for operation: %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=8
[tryRouteDataMove] Routing from Tile#1 @t=5 to Tile#1 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #32
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#1 @t=8
[HeuristicMapping] Successfully mapped operation %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 123 candidate locations for operation: %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#6 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #193
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#6 @t=9
[HeuristicMapping] Successfully mapped operation %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 113 candidate locations for operation: %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=7 to Tile#6 @t=10
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#6 @t=10
[HeuristicMapping] Successfully mapped operation %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 116 candidate locations for operation: %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=8 to Tile#10 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #322
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#10 @t=10
[HeuristicMapping] Successfully mapped operation %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 120 candidate locations for operation: %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=9
[tryRouteDataMove] Routing from Tile#9 @t=8 to Tile#13 @t=9
[tryRouteDataMove] Routing from Tile#9 @t=8 to Tile#13 @t=9
[tryRouteDataMove] Cannot find routing path from Tile#9 @t=8 to Tile#13 @t=9
[HeuristicMapping] Failed to map operation %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1> to candidate location 1/1
[HeuristicMapping] Found 120 candidate locations for operation: %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[HeuristicMapping] All 1 locations for 56 tried, backtracking...
[HeuristicMapping] Backtracking to operation 55 (depth = 1).
[HeuristicMapping] Found 116 candidate locations for operation: %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] All 1 locations for 55 tried, backtracking...
[HeuristicMapping] Backtracking to operation 54 (depth = 2).
[HeuristicMapping] Max backtrack depth exceeded: 2 > 1.
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 228 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 152 non-materialized operations, 76 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (level: 0)
1 %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1> (level: 0)
2 %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
3 %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
4 %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1> (level: 1)
5 %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
6 %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
7 %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
8 %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
9 %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
10 %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
12 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
13 %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 4)
14 %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 4)
15 %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
16 %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
17 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
18 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
19 %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 5)
20 %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
21 %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
22 %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
23 %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
24 %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
25 %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
26 %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
27 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
28 %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
29 %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
30 %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
31 %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
32 %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
33 %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 6)
34 %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
35 %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
36 %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
37 %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 7)
38 %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
39 %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
40 %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 7)
41 %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
42 %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
43 %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
44 %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
45 %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
46 %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
47 %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 8)
48 %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
49 %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
50 %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
51 %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
52 %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
53 %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
54 %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
55 %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
56 %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1> (level: 8)
57 %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
58 %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
59 %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
60 %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
61 %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
62 %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
63 %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
64 %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
65 %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
66 %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
67 %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
68 %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
69 %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
70 %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
71 %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
72 %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
73 "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
74 "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
75 neura.return_void %204 : !neura.data<i1, i1> (level: 9)
[HeuristicMapping] Found 208 candidate locations for operation: %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=0
[HeuristicMapping] Successfully mapped operation %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 207 candidate locations for operation: %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=0
[HeuristicMapping] Successfully mapped operation %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 173 candidate locations for operation: %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=1
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#4 @t=1
[HeuristicMapping] Successfully mapped operation %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 180 candidate locations for operation: %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=1
[tryRouteDataMove] Routing from Tile#4 @t=0 to Tile#5 @t=1
[HeuristicMapping] Successfully mapped operation %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 204 candidate locations for operation: %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=1
[HeuristicMapping] Successfully mapped operation %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 178 candidate locations for operation: %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=2
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#5 @t=2
[HeuristicMapping] Successfully mapped operation %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 185 candidate locations for operation: %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=2
[tryRouteDataMove] Routing from Tile#5 @t=1 to Tile#9 @t=2
[HeuristicMapping] Successfully mapped operation %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 177 candidate locations for operation: %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=2
[tryRouteDataMove] Routing from Tile#8 @t=1 to Tile#8 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #256
[HeuristicMapping] Successfully mapped operation %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 189 candidate locations for operation: %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=3
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#4 @t=3
[HeuristicMapping] Successfully mapped operation %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 174 candidate locations for operation: %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=3
[tryRouteDataMove] Routing from Tile#8 @t=2 to Tile#8 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #256
[HeuristicMapping] Successfully mapped operation %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 176 candidate locations for operation: %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=3
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#9 @t=2 to Tile#5 @t=3
[HeuristicMapping] Successfully mapped operation %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 197 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=3
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 196 candidate locations for operation: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#3 @t=3
[HeuristicMapping] Successfully mapped operation %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 167 candidate locations for operation: %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=4
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#9 @t=4
[tryRouteDataMove] Routing from Tile#8 @t=3 to Tile#9 @t=4
[HeuristicMapping] Successfully mapped operation %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 170 candidate locations for operation: %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=4
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#4 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #128
[HeuristicMapping] Successfully mapped operation %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 177 candidate locations for operation: %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=4
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#10 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 162 candidate locations for operation: %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=4
[tryRouteDataMove] Routing from Tile#3 @t=3 to Tile#2 @t=4
[HeuristicMapping] Successfully mapped operation %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 191 candidate locations for operation: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=4
[HeuristicMapping] Successfully mapped operation %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 190 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=4
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 172 candidate locations for operation: %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=5
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#9 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #288
[HeuristicMapping] Successfully mapped operation %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 171 candidate locations for operation: %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=5
[tryRouteDataMove] Routing from Tile#10 @t=4 to Tile#10 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 164 candidate locations for operation: %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=5
[tryRouteDataMove] Routing from Tile#2 @t=4 to Tile#2 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #64
[HeuristicMapping] Successfully mapped operation %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 156 candidate locations for operation: %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=5
[tryRouteDataMove] Routing from Tile#0 @t=4 to Tile#4 @t=5
[HeuristicMapping] Successfully mapped operation %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 169 candidate locations for operation: %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=5
[tryRouteDataMove] Routing from Tile#8 @t=3 to Tile#8 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #256
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#8 @t=5
[HeuristicMapping] Successfully mapped operation %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 169 candidate locations for operation: %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=6
[tryRouteDataMove] Routing from Tile#9 @t=2 to Tile#9 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #289
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#9 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #290
[HeuristicMapping] Successfully mapped operation %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 162 candidate locations for operation: %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#11 @t=5
[tryRouteDataMove] Routing from Tile#11 @t=4 to Tile#11 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #352
[HeuristicMapping] Successfully mapped operation %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 159 candidate locations for operation: %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=5
[tryRouteDataMove] Routing from Tile#4 @t=4 to Tile#5 @t=5
[tryRouteDataMove] Routing from Tile#9 @t=4 to Tile#5 @t=5
[HeuristicMapping] Successfully mapped operation %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 181 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=5
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 156 candidate locations for operation: %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=6
[tryRouteDataMove] Routing from Tile#11 @t=5 to Tile#10 @t=6
[HeuristicMapping] Successfully mapped operation %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 154 candidate locations for operation: %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=7
[tryRouteDataMove] Routing from Tile#10 @t=5 to Tile#5 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=4 to Tile#5 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #161
[HeuristicMapping] Successfully mapped operation %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 147 candidate locations for operation: %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=6
[tryRouteDataMove] Routing from Tile#0 @t=5 to Tile#4 @t=6
[HeuristicMapping] Successfully mapped operation %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 151 candidate locations for operation: %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=5 to Tile#4 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #128
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#4 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #129
[HeuristicMapping] Successfully mapped operation %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 155 candidate locations for operation: %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=6
[tryRouteDataMove] Routing from Tile#2 @t=5 to Tile#2 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #64
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#2 @t=6
[HeuristicMapping] Successfully mapped operation %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 158 candidate locations for operation: %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=5 to Tile#9 @t=7
[tryRouteDataMove] Routing from Tile#9 @t=6 to Tile#9 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #289
[HeuristicMapping] Successfully mapped operation %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 105 candidate locations for operation: %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=6
[tryRouteDataMove] Routing from Tile#8 @t=3 to Tile#8 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #257
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#8 @t=6
[tryRouteDataMove] Routing from Tile#8 @t=6 to Tile#8 @t=16
[tryRouteDataMove] Successfully routed on same tile using Register #257
[HeuristicMapping] Successfully mapped operation %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 100 candidate locations for operation: %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=2 to Tile#9 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #291
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#9 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #292
[tryRouteDataMove] Routing from Tile#9 @t=8 to Tile#9 @t=15
[tryRouteDataMove] Successfully routed on same tile using Register #288
[HeuristicMapping] Successfully mapped operation %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 99 candidate locations for operation: %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#5 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #160
[tryRouteDataMove] Routing from Tile#9 @t=5 to Tile#5 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#5 @t=15
[tryRouteDataMove] Successfully routed on same tile using Register #162
[HeuristicMapping] Successfully mapped operation %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 154 candidate locations for operation: %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=8
[tryRouteDataMove] Routing from Tile#9 @t=7 to Tile#8 @t=8
[tryRouteDataMove] Routing from Tile#8 @t=5 to Tile#8 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #256
[HeuristicMapping] Successfully mapped operation %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 155 candidate locations for operation: %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=8
[tryRouteDataMove] Routing from Tile#4 @t=5 to Tile#5 @t=8
[HeuristicMapping] Successfully mapped operation %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 143 candidate locations for operation: %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=8
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#4 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #130
[HeuristicMapping] Successfully mapped operation %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 168 candidate locations for operation: %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#6 @t=7
[HeuristicMapping] Successfully mapped operation %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 142 candidate locations for operation: %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#0 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#0 @t=7
[HeuristicMapping] Successfully mapped operation %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 153 candidate locations for operation: %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=7
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#10 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #320
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#10 @t=7
[HeuristicMapping] Successfully mapped operation %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 141 candidate locations for operation: %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=8
[tryRouteDataMove] Routing from Tile#4 @t=7 to Tile#0 @t=8
[HeuristicMapping] Successfully mapped operation %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 147 candidate locations for operation: %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=7
[tryRouteDataMove] Routing from Tile#2 @t=6 to Tile#2 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #64
[HeuristicMapping] Successfully mapped operation %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 148 candidate locations for operation: %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=7 to Tile#6 @t=8
[HeuristicMapping] Successfully mapped operation %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 146 candidate locations for operation: %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=7 to Tile#1 @t=8
[HeuristicMapping] Successfully mapped operation %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 136 candidate locations for operation: %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=9
[tryRouteDataMove] Routing from Tile#8 @t=8 to Tile#8 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #256
[HeuristicMapping] Successfully mapped operation %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 122 candidate locations for operation: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=7 to Tile#5 @t=10
[tryRouteDataMove] Routing from Tile#6 @t=8 to Tile#5 @t=10
[tryRouteDataMove] Routing from Tile#0 @t=8 to Tile#5 @t=10
[HeuristicMapping] Successfully mapped operation %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 135 candidate locations for operation: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=9
[tryRouteDataMove] Routing from Tile#1 @t=8 to Tile#0 @t=9
[tryRouteDataMove] Routing from Tile#0 @t=7 to Tile#0 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #0
[tryRouteDataMove] Routing from Tile#2 @t=7 to Tile#0 @t=9
[tryRouteDataMove] Cannot find routing path from Tile#2 @t=7 to Tile#0 @t=9
[HeuristicMapping] Failed to map operation %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> to candidate location 1/1
[HeuristicMapping] Found 135 candidate locations for operation: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] All 1 locations for 49 tried, backtracking...
[HeuristicMapping] Backtracking to operation 48 (depth = 1).
[HeuristicMapping] Found 122 candidate locations for operation: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] All 1 locations for 48 tried, backtracking...
[HeuristicMapping] Backtracking to operation 47 (depth = 2).
[HeuristicMapping] Max backtrack depth exceeded: 2 > 1.
---------------------------------------------------------
[HeuristicMapping] Starting mapping with 228 operations.
Configuration: MAX Backtrack Depth = 1, MAX Candidate Locations = 1
[HeuristicMapping] Filtered 152 non-materialized operations, 76 operations require physical mapping.
[HeuristicMapping] Materialized operations list:
0 %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1> (level: 0)
1 %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1> (level: 0)
2 %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
3 %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 1)
4 %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1> (level: 1)
5 %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
6 %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
7 %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 2)
8 %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
9 %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 3)
10 %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 3)
11 %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
12 %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1> (level: 3)
13 %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 4)
14 %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 4)
15 %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
16 %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 4)
17 %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
18 %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1> (level: 4)
19 %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 5)
20 %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
21 %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
22 %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
23 %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
24 %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
25 %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 5)
26 %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 5)
27 %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1> (level: 5)
28 %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
29 %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
30 %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 6)
31 %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
32 %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1> (level: 6)
33 %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1> (level: 6)
34 %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
35 %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
36 %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 6)
37 %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1> (level: 7)
38 %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
39 %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1> (level: 7)
40 %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1> (level: 7)
41 %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
42 %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1> (level: 7)
43 %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
44 %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
45 %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
46 %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1> (level: 7)
47 %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1> (level: 8)
48 %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
49 %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1> (level: 8)
50 %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
51 %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
52 %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
53 %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 8)
54 %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
55 %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 8)
56 %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1> (level: 8)
57 %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
58 %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
59 %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
60 %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
61 %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
62 %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
63 %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
64 %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
65 %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
66 %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
67 %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
68 %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
69 %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
70 %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
71 %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1> (level: 9)
72 %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1> (level: 9)
73 "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
74 "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> () (level: 9)
75 neura.return_void %204 : !neura.data<i1, i1> (level: 9)
[HeuristicMapping] Found 224 candidate locations for operation: %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=0
[HeuristicMapping] Successfully mapped operation %5 = "neura.grant_once"() <{constant_value = 0 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 223 candidate locations for operation: %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=0
[HeuristicMapping] Successfully mapped operation %6 = "neura.grant_once"() <{constant_value = 1 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 189 candidate locations for operation: %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=1
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#0 @t=1
[tryRouteDataMove] Successfully routed on same tile using Register #0
[HeuristicMapping] Successfully mapped operation %25 = neura.phi_start %24, %23 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 196 candidate locations for operation: %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=1
[tryRouteDataMove] Routing from Tile#4 @t=0 to Tile#4 @t=1
[tryRouteDataMove] Successfully routed on same tile using Register #128
[HeuristicMapping] Successfully mapped operation %13 = neura.phi_start %12, %11 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 220 candidate locations for operation: %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=1
[HeuristicMapping] Successfully mapped operation %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> : () -> !neura.data<i64, i1>
[HeuristicMapping] Found 186 candidate locations for operation: %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=2
[tryRouteDataMove] Routing from Tile#0 @t=1 to Tile#4 @t=2
[HeuristicMapping] Successfully mapped operation %67 = neura.phi_start %66, %65 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 193 candidate locations for operation: %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=2
[tryRouteDataMove] Routing from Tile#4 @t=1 to Tile#5 @t=2
[HeuristicMapping] Successfully mapped operation %55 = neura.phi_start %54, %53 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 202 candidate locations for operation: %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=2
[tryRouteDataMove] Routing from Tile#10 @t=1 to Tile#10 @t=2
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %10 = neura.phi_start %9, %8 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 206 candidate locations for operation: %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=3
[tryRouteDataMove] Routing from Tile#0 @t=0 to Tile#5 @t=3
[HeuristicMapping] Successfully mapped operation %34 = neura.phi_start %33, %32 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 199 candidate locations for operation: %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=3
[tryRouteDataMove] Routing from Tile#10 @t=2 to Tile#10 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #320
[HeuristicMapping] Successfully mapped operation %52 = neura.phi_start %51, %50 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 187 candidate locations for operation: %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=3
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#4 @t=3
[tryRouteDataMove] Successfully routed on same tile using Register #128
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#4 @t=3
[HeuristicMapping] Successfully mapped operation %103 = "neura.add"(%101, %102) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 213 candidate locations for operation: %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#3 @t=3
[HeuristicMapping] Successfully mapped operation %0 = "neura.grant_once"() <{constant_value = "%arg0"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 212 candidate locations for operation: %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#12 @t=3
[HeuristicMapping] Successfully mapped operation %1 = "neura.grant_once"() <{constant_value = "%arg1"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 177 candidate locations for operation: %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=5
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#6 @t=5
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#6 @t=5
[HeuristicMapping] Successfully mapped operation %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 193 candidate locations for operation: %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=4
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#5 @t=4
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %49 = neura.phi_start %48, %47 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 177 candidate locations for operation: %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=4
[tryRouteDataMove] Routing from Tile#3 @t=3 to Tile#2 @t=4
[HeuristicMapping] Successfully mapped operation %19 = neura.phi_start %18, %17 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 178 candidate locations for operation: %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=4
[tryRouteDataMove] Routing from Tile#12 @t=3 to Tile#13 @t=4
[HeuristicMapping] Successfully mapped operation %22 = neura.phi_start %21, %20 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 207 candidate locations for operation: %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=4
[HeuristicMapping] Successfully mapped operation %2 = "neura.grant_once"() <{constant_value = "%arg2"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 206 candidate locations for operation: %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#12 @t=4
[HeuristicMapping] Successfully mapped operation %3 = "neura.grant_once"() <{constant_value = "%arg3"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 188 candidate locations for operation: %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=6
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#6 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #192
[HeuristicMapping] Successfully mapped operation %108 = "neura.not"(%107) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 179 candidate locations for operation: %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=5
[tryRouteDataMove] Routing from Tile#2 @t=4 to Tile#2 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #64
[HeuristicMapping] Successfully mapped operation %61 = neura.phi_start %60, %59 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 181 candidate locations for operation: %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#13 @t=5
[tryRouteDataMove] Routing from Tile#13 @t=4 to Tile#13 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #416
[HeuristicMapping] Successfully mapped operation %64 = neura.phi_start %63, %62 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 179 candidate locations for operation: %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=5
[tryRouteDataMove] Routing from Tile#14 @t=4 to Tile#14 @t=5
[tryRouteDataMove] Successfully routed on same tile using Register #448
[HeuristicMapping] Successfully mapped operation %28 = neura.phi_start %27, %26 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 184 candidate locations for operation: %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=6
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#10 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #320
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#10 @t=6
[HeuristicMapping] Successfully mapped operation %147 = neura.grant_predicate %145, %146 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 182 candidate locations for operation: %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #161
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#5 @t=6
[HeuristicMapping] Successfully mapped operation %144 = neura.grant_predicate %142, %143 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 172 candidate locations for operation: %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=5
[tryRouteDataMove] Routing from Tile#12 @t=4 to Tile#8 @t=5
[HeuristicMapping] Successfully mapped operation %16 = neura.phi_start %15, %14 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 174 candidate locations for operation: %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#6 @t=7
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#6 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #193
[HeuristicMapping] Successfully mapped operation %141 = neura.grant_predicate %139, %140 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 197 candidate locations for operation: %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=5
[HeuristicMapping] Successfully mapped operation %4 = "neura.grant_once"() <{constant_value = "%arg4"}> : () -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 175 candidate locations for operation: %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=6
[tryRouteDataMove] Routing from Tile#8 @t=5 to Tile#8 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #256
[HeuristicMapping] Successfully mapped operation %58 = neura.phi_start %57, %56 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 171 candidate locations for operation: %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=6
[tryRouteDataMove] Routing from Tile#2 @t=5 to Tile#1 @t=6
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#1 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#1 @t=6
[HeuristicMapping] Successfully mapped operation %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 172 candidate locations for operation: %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=5 to Tile#4 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #128
[HeuristicMapping] Successfully mapped operation %31 = neura.phi_start %30, %29 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 172 candidate locations for operation: %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=6
[tryRouteDataMove] Routing from Tile#14 @t=5 to Tile#14 @t=6
[tryRouteDataMove] Successfully routed on same tile using Register #448
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#14 @t=6
[HeuristicMapping] Successfully mapped operation %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 174 candidate locations for operation: %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#12 @t=6
[tryRouteDataMove] Routing from Tile#13 @t=5 to Tile#12 @t=6
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#12 @t=6
[HeuristicMapping] Successfully mapped operation %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 174 candidate locations for operation: %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=7 to Tile#6 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #192
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#6 @t=8
[HeuristicMapping] Successfully mapped operation %168 = "neura.add"(%166, %167) : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
[HeuristicMapping] Found 121 candidate locations for operation: %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=7
[tryRouteDataMove] Routing from Tile#10 @t=3 to Tile#10 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #321
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#10 @t=7
[tryRouteDataMove] Routing from Tile#10 @t=7 to Tile#10 @t=17
[tryRouteDataMove] Successfully routed on same tile using Register #321
[HeuristicMapping] Successfully mapped operation %126 = neura.grant_predicate %124, %125 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 102 candidate locations for operation: %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=2 to Tile#5 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #162
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#5 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=7 to Tile#5 @t=16
[tryRouteDataMove] Successfully routed on same tile using Register #160
[HeuristicMapping] Successfully mapped operation %123 = neura.grant_predicate %121, %122 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 87 candidate locations for operation: %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=8
[tryRouteDataMove] Routing from Tile#4 @t=3 to Tile#5 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#5 @t=8
[tryRouteDataMove] Routing from Tile#5 @t=8 to Tile#4 @t=16
[HeuristicMapping] Successfully mapped operation %111 = neura.grant_predicate %109, %110 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 170 candidate locations for operation: %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=8 to Tile#6 @t=9
[tryRouteDataMove] Successfully routed on same tile using Register #192
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#6 @t=9
[HeuristicMapping] Successfully mapped operation %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 173 candidate locations for operation: %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#14 @t=7
[tryRouteDataMove] Routing from Tile#14 @t=5 to Tile#14 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #449
[HeuristicMapping] Successfully mapped operation %43 = neura.phi_start %42, %41 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 163 candidate locations for operation: %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#4 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #128
[HeuristicMapping] Successfully mapped operation %46 = neura.phi_start %45, %44 : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 184 candidate locations for operation: %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#0 @t=7
[tryRouteDataMove] Routing from Tile#0 @t=1 to Tile#0 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #0
[HeuristicMapping] Successfully mapped operation %40 = neura.phi_start %39, %38 : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 161 candidate locations for operation: %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=7
[tryRouteDataMove] Routing from Tile#4 @t=6 to Tile#8 @t=7
[tryRouteDataMove] Routing from Tile#5 @t=3 to Tile#8 @t=7
[HeuristicMapping] Successfully mapped operation %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 163 candidate locations for operation: %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=8
[tryRouteDataMove] Routing from Tile#8 @t=6 to Tile#8 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #256
[tryRouteDataMove] Routing from Tile#4 @t=2 to Tile#8 @t=8
[HeuristicMapping] Successfully mapped operation %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> : !neura.data<i32, i1>
[HeuristicMapping] Found 163 candidate locations for operation: %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#15 @t=7
[tryRouteDataMove] Routing from Tile#14 @t=6 to Tile#15 @t=7
[HeuristicMapping] Successfully mapped operation %89 = "neura.load"(%88) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 157 candidate locations for operation: %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#12 @t=7
[tryRouteDataMove] Routing from Tile#12 @t=6 to Tile#12 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #384
[HeuristicMapping] Successfully mapped operation %72 = "neura.load"(%71) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 162 candidate locations for operation: %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=7
[tryRouteDataMove] Routing from Tile#1 @t=6 to Tile#1 @t=7
[tryRouteDataMove] Successfully routed on same tile using Register #32
[HeuristicMapping] Successfully mapped operation %91 = "neura.load"(%90) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 161 candidate locations for operation: %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=7
[tryRouteDataMove] Routing from Tile#1 @t=6 to Tile#2 @t=7
[HeuristicMapping] Successfully mapped operation %81 = "neura.load"(%80) : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 160 candidate locations for operation: %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=10
[tryRouteDataMove] Routing from Tile#6 @t=9 to Tile#6 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #192
[HeuristicMapping] Successfully mapped operation %173 = "neura.not"(%172) : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
[HeuristicMapping] Found 140 candidate locations for operation: %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=10
[tryRouteDataMove] Routing from Tile#8 @t=8 to Tile#9 @t=10
[tryRouteDataMove] Routing from Tile#1 @t=7 to Tile#9 @t=10
[tryRouteDataMove] Routing from Tile#15 @t=7 to Tile#9 @t=10
[HeuristicMapping] Successfully mapped operation %98 = "neura.mul_add"(%95, %96, %97) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 133 candidate locations for operation: %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#4 @t=10
[tryRouteDataMove] Routing from Tile#2 @t=7 to Tile#4 @t=10
[tryRouteDataMove] Routing from Tile#8 @t=7 to Tile#4 @t=10
[tryRouteDataMove] Routing from Tile#12 @t=7 to Tile#4 @t=10
[HeuristicMapping] Successfully mapped operation %85 = "neura.mul_add"(%82, %83, %84) : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
[HeuristicMapping] Found 154 candidate locations for operation: %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=8
[tryRouteDataMove] Routing from Tile#8 @t=6 to Tile#10 @t=8
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#10 @t=8
[HeuristicMapping] Successfully mapped operation %165 = neura.grant_predicate %163, %164 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 160 candidate locations for operation: %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=8
[tryRouteDataMove] Routing from Tile#2 @t=5 to Tile#2 @t=8
[tryRouteDataMove] Successfully routed on same tile using Register #64
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#2 @t=8
[HeuristicMapping] Successfully mapped operation %162 = neura.grant_predicate %160, %161 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 150 candidate locations for operation: %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=9
[tryRouteDataMove] Routing from Tile#13 @t=5 to Tile#5 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#5 @t=9
[HeuristicMapping] Successfully mapped operation %159 = neura.grant_predicate %157, %158 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 135 candidate locations for operation: %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=9
[tryRouteDataMove] Routing from Tile#0 @t=7 to Tile#2 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#2 @t=9
[HeuristicMapping] Successfully mapped operation %156 = neura.grant_predicate %154, %155 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 140 candidate locations for operation: %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=9
[tryRouteDataMove] Routing from Tile#14 @t=7 to Tile#10 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#10 @t=9
[HeuristicMapping] Successfully mapped operation %153 = neura.grant_predicate %151, %152 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 135 candidate locations for operation: %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=10
[tryRouteDataMove] Routing from Tile#4 @t=7 to Tile#5 @t=10
[tryRouteDataMove] Routing from Tile#6 @t=5 to Tile#5 @t=10
[HeuristicMapping] Successfully mapped operation %150 = neura.grant_predicate %148, %149 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 152 candidate locations for operation: %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=11
[tryRouteDataMove] Routing from Tile#6 @t=9 to Tile#6 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #193
[tryRouteDataMove] Routing from Tile#6 @t=9 to Tile#6 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #194
[HeuristicMapping] Successfully mapped operation %203 = neura.grant_predicate %201, %202 : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
[HeuristicMapping] Found 100 candidate locations for operation: %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#1 @t=9
[tryRouteDataMove] Routing from Tile#0 @t=7 to Tile#1 @t=9
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#1 @t=9
[tryRouteDataMove] Routing from Tile#1 @t=9 to Tile#0 @t=21
[HeuristicMapping] Successfully mapped operation %138 = neura.grant_predicate %136, %137 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 106 candidate locations for operation: %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=10
[tryRouteDataMove] Routing from Tile#14 @t=7 to Tile#10 @t=10
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#10 @t=10
[tryRouteDataMove] Routing from Tile#10 @t=10 to Tile#14 @t=21
[HeuristicMapping] Successfully mapped operation %135 = neura.grant_predicate %133, %134 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 98 candidate locations for operation: %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=11
[tryRouteDataMove] Routing from Tile#4 @t=7 to Tile#5 @t=11
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#5 @t=11
[tryRouteDataMove] Routing from Tile#5 @t=11 to Tile#4 @t=21
[HeuristicMapping] Successfully mapped operation %132 = neura.grant_predicate %130, %131 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 76 candidate locations for operation: %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=12
[tryRouteDataMove] Routing from Tile#5 @t=4 to Tile#5 @t=12
[tryRouteDataMove] Successfully routed on same tile using Register #164
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#5 @t=12
[tryRouteDataMove] Routing from Tile#5 @t=12 to Tile#5 @t=18
[tryRouteDataMove] Successfully routed on same tile using Register #163
[HeuristicMapping] Successfully mapped operation %129 = neura.grant_predicate %127, %128 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 85 candidate locations for operation: %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=11
[tryRouteDataMove] Routing from Tile#8 @t=6 to Tile#10 @t=11
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#10 @t=11
[tryRouteDataMove] Routing from Tile#10 @t=11 to Tile#8 @t=20
[HeuristicMapping] Successfully mapped operation %120 = neura.grant_predicate %118, %119 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 68 candidate locations for operation: %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=10
[tryRouteDataMove] Routing from Tile#2 @t=5 to Tile#2 @t=10
[tryRouteDataMove] Successfully routed on same tile using Register #67
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#2 @t=10
[tryRouteDataMove] Routing from Tile#2 @t=10 to Tile#2 @t=19
[tryRouteDataMove] Successfully routed on same tile using Register #65
[HeuristicMapping] Successfully mapped operation %117 = neura.grant_predicate %115, %116 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 66 candidate locations for operation: %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=12
[tryRouteDataMove] Routing from Tile#13 @t=5 to Tile#10 @t=12
[tryRouteDataMove] Routing from Tile#6 @t=6 to Tile#10 @t=12
[tryRouteDataMove] Routing from Tile#10 @t=12 to Tile#13 @t=19
[HeuristicMapping] Successfully mapped operation %114 = neura.grant_predicate %112, %113 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 33 candidate locations for operation: %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=12
[tryRouteDataMove] Routing from Tile#10 @t=6 to Tile#6 @t=12
[tryRouteDataMove] Routing from Tile#6 @t=10 to Tile#6 @t=12
[tryRouteDataMove] Successfully routed on same tile using Register #192
[tryRouteDataMove] Routing from Tile#6 @t=12 to Tile#10 @t=16
[HeuristicMapping] Successfully mapped operation %200 = neura.grant_predicate %198, %199 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 13 candidate locations for operation: %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=13
[tryRouteDataMove] Routing from Tile#5 @t=6 to Tile#5 @t=13
[tryRouteDataMove] Successfully routed on same tile using Register #165
[tryRouteDataMove] Routing from Tile#6 @t=10 to Tile#5 @t=13
[tryRouteDataMove] Routing from Tile#5 @t=13 to Tile#4 @t=15
[HeuristicMapping] Successfully mapped operation %197 = neura.grant_predicate %195, %196 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 51 candidate locations for operation: %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=13
[tryRouteDataMove] Routing from Tile#10 @t=8 to Tile#10 @t=13
[tryRouteDataMove] Successfully routed on same tile using Register #324
[tryRouteDataMove] Routing from Tile#6 @t=10 to Tile#10 @t=13
[tryRouteDataMove] Routing from Tile#10 @t=13 to Tile#8 @t=19
[HeuristicMapping] Successfully mapped operation %194 = neura.grant_predicate %192, %193 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 40 candidate locations for operation: %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=11
[tryRouteDataMove] Routing from Tile#2 @t=8 to Tile#2 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #64
[tryRouteDataMove] Routing from Tile#6 @t=10 to Tile#2 @t=11
[tryRouteDataMove] Routing from Tile#2 @t=11 to Tile#2 @t=18
[tryRouteDataMove] Successfully routed on same tile using Register #64
[HeuristicMapping] Successfully mapped operation %191 = neura.grant_predicate %189, %190 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 33 candidate locations for operation: %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=13
[tryRouteDataMove] Routing from Tile#5 @t=9 to Tile#6 @t=13
[tryRouteDataMove] Routing from Tile#6 @t=10 to Tile#6 @t=13
[tryRouteDataMove] Successfully routed on same tile using Register #197
[tryRouteDataMove] Routing from Tile#6 @t=13 to Tile#13 @t=18
[HeuristicMapping] Successfully mapped operation %188 = neura.grant_predicate %186, %187 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 4 candidate locations for operation: %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=12
[tryRouteDataMove] Routing from Tile#2 @t=9 to Tile#2 @t=12
[tryRouteDataMove] Successfully routed on same tile using Register #66
[tryRouteDataMove] Routing from Tile#6 @t=10 to Tile#2 @t=12
[tryRouteDataMove] Routing from Tile#2 @t=12 to Tile#0 @t=15
[HeuristicMapping] Successfully mapped operation %185 = neura.grant_predicate %183, %184 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 44 candidate locations for operation: %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#10 @t=14
[tryRouteDataMove] Routing from Tile#10 @t=9 to Tile#10 @t=14
[tryRouteDataMove] Successfully routed on same tile using Register #325
[tryRouteDataMove] Routing from Tile#6 @t=10 to Tile#10 @t=14
[tryRouteDataMove] Routing from Tile#10 @t=14 to Tile#14 @t=19
[HeuristicMapping] Successfully mapped operation %182 = neura.grant_predicate %180, %181 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 48 candidate locations for operation: %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#5 @t=14
[tryRouteDataMove] Routing from Tile#5 @t=10 to Tile#5 @t=14
[tryRouteDataMove] Successfully routed on same tile using Register #162
[tryRouteDataMove] Routing from Tile#6 @t=10 to Tile#5 @t=14
[tryRouteDataMove] Routing from Tile#5 @t=14 to Tile#4 @t=20
[HeuristicMapping] Successfully mapped operation %179 = neura.grant_predicate %177, %178 : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
[HeuristicMapping] Found 16 candidate locations for operation: %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#6 @t=14
[tryRouteDataMove] Routing from Tile#6 @t=8 to Tile#6 @t=14
[tryRouteDataMove] Successfully routed on same tile using Register #202
[tryRouteDataMove] Routing from Tile#6 @t=10 to Tile#6 @t=14
[tryRouteDataMove] Successfully routed on same tile using Register #203
[tryRouteDataMove] Routing from Tile#6 @t=14 to Tile#5 @t=17
[HeuristicMapping] Successfully mapped operation %176 = neura.grant_predicate %174, %175 : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
[HeuristicMapping] Found 137 candidate locations for operation: "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[HeuristicMapping] Trying candidate 1/1 at tile#9 @t=11
[tryRouteDataMove] Routing from Tile#9 @t=10 to Tile#9 @t=11
[tryRouteDataMove] Successfully routed on same tile using Register #288
[tryRouteDataMove] Routing from Tile#14 @t=6 to Tile#9 @t=11
[HeuristicMapping] Successfully mapped operation "neura.store"(%99, %100) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[HeuristicMapping] Found 131 candidate locations for operation: "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[HeuristicMapping] Trying candidate 1/1 at tile#8 @t=11
[tryRouteDataMove] Routing from Tile#4 @t=10 to Tile#8 @t=11
[tryRouteDataMove] Routing from Tile#12 @t=6 to Tile#8 @t=11
[HeuristicMapping] Successfully mapped operation "neura.store"(%86, %87) : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
[HeuristicMapping] Found 112 candidate locations for operation: neura.return_void %204 : !neura.data<i1, i1>
[HeuristicMapping] Trying candidate 1/1 at tile#2 @t=13
[tryRouteDataMove] Routing from Tile#6 @t=11 to Tile#2 @t=13
[HeuristicMapping] Successfully mapped operation neura.return_void %204 : !neura.data<i1, i1>
[HeuristicMapping] Successfully mapped all 76 operations.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, llvm.ident = "clang version 20.1.7 (https://github.com/llvm/llvm-project.git 6146a88f60492b520a36f8f8f3231e15f3cc6082)"} {
  llvm.mlir.global external local_unnamed_addr @A(dense<0> : tensor<1024x1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x array<1024 x i32>>
  llvm.mlir.global external local_unnamed_addr @s(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  llvm.mlir.global external local_unnamed_addr @q(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  llvm.mlir.global external local_unnamed_addr @p(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  llvm.mlir.global external local_unnamed_addr @r(dense<0> : tensor<1024xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<1024 x i32>
  func.func @_Z6kernelPA1024_iPiS1_S1_S1_(%arg0: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg1: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg2: !llvm.ptr {llvm.nocapture, llvm.noundef}, %arg3: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}, %arg4: !llvm.ptr {llvm.nocapture, llvm.noundef, llvm.readonly}) -> !llvm.void attributes {CConv = #llvm.cconv<ccc>, accelerator = "neura", dataflow_mode = "predicate", linkage = #llvm.linkage<external>, mapping_info = {compiled_ii = 14 : i32, mapping_mode = "spatial-temporal", mapping_strategy = "heuristic", rec_mii = 9 : i32, res_mii = 5 : i32, x_tiles = 4 : i32, y_tiles = 4 : i32}, memory_effects = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", unnamed_addr = 1 : i64, visibility_ = 0 : i64} {
    %0 = "neura.grant_once"() <{constant_value = "%arg0"}> {dfg_id = 0 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 3 : i32, y = 0 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
    %1 = "neura.grant_once"() <{constant_value = "%arg1"}> {dfg_id = 1 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 3 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
    %2 = "neura.grant_once"() <{constant_value = "%arg2"}> {dfg_id = 2 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 3 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
    %3 = "neura.grant_once"() <{constant_value = "%arg3"}> {dfg_id = 3 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 0 : i32, y = 3 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
    %4 = "neura.grant_once"() <{constant_value = "%arg4"}> {dfg_id = 4 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<!llvm.ptr, i1>
    %5 = "neura.grant_once"() <{constant_value = 0 : i64}> {dfg_id = 5 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 0 : i32}]} : () -> !neura.data<i64, i1>
    %6 = "neura.grant_once"() <{constant_value = 1 : i64}> {dfg_id = 6 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 0 : i32, x = 0 : i32, y = 1 : i32}]} : () -> !neura.data<i64, i1>
    %7 = "neura.grant_once"() <{constant_value = 1024 : i64}> {dfg_id = 7 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 2 : i32, y = 2 : i32}]} : () -> !neura.data<i64, i1>
    %8 = neura.reserve {dfg_id = 8 : i32} : !neura.data<i64, i1>
    %9 = "neura.data_mov"(%7) {dfg_id = 36 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %10 = neura.phi_start %9, %8 {dfg_id = 45 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %11 = neura.reserve {dfg_id = 9 : i32} : !neura.data<i64, i1>
    %12 = "neura.data_mov"(%6) {dfg_id = 35 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %13 = neura.phi_start %12, %11 {dfg_id = 44 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %14 = neura.reserve {dfg_id = 10 : i32} : !neura.data<!llvm.ptr, i1>
    %15 = "neura.data_mov"(%3) {dfg_id = 31 : i32, mapping_locs = [{id = 39 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %16 = neura.phi_start %15, %14 {dfg_id = 40 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
    %17 = neura.reserve {dfg_id = 11 : i32} : !neura.data<!llvm.ptr, i1>
    %18 = "neura.data_mov"(%0) {dfg_id = 28 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %19 = neura.phi_start %18, %17 {dfg_id = 37 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
    %20 = neura.reserve {dfg_id = 12 : i32} : !neura.data<!llvm.ptr, i1>
    %21 = "neura.data_mov"(%1) {dfg_id = 29 : i32, mapping_locs = [{id = 38 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %22 = neura.phi_start %21, %20 {dfg_id = 38 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 3 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
    %23 = neura.reserve {dfg_id = 13 : i32} : !neura.data<i64, i1>
    %24 = "neura.data_mov"(%5) {dfg_id = 33 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 0 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %25 = neura.phi_start %24, %23 {dfg_id = 42 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 1 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %26 = neura.reserve {dfg_id = 14 : i32} : !neura.data<!llvm.ptr, i1>
    %27 = "neura.data_mov"(%2) {dfg_id = 30 : i32, mapping_locs = [{id = 448 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %28 = neura.phi_start %27, %26 {dfg_id = 39 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 3 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
    %29 = neura.reserve {dfg_id = 15 : i32} : !neura.data<!llvm.ptr, i1>
    %30 = "neura.data_mov"(%4) {dfg_id = 32 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %31 = neura.phi_start %30, %29 {dfg_id = 41 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
    %32 = neura.reserve {dfg_id = 16 : i32} : !neura.data<i64, i1>
    %33 = "neura.data_mov"(%5) {dfg_id = 34 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 0 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 0 : i32}, {id = 4 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}, {id = 160 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %34 = neura.phi_start %33, %32 {dfg_id = 43 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %35 = "neura.data_mov"(%28) {dfg_id = 49 : i32, mapping_locs = [{id = 448 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %36 = "neura.data_mov"(%34) {dfg_id = 57 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 20 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 34 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %37 = "neura.gep"(%35, %36) <{operandSegmentSizes = array<i32: 1, 1>}> {dfg_id = 69 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
    %38 = neura.reserve {dfg_id = 17 : i32} : !neura.data<i64, i1>
    %39 = "neura.data_mov"(%25) {dfg_id = 54 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 1 : i32}, {id = 0 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}, {id = 0 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 0 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 0 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 0 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %40 = neura.phi_start %39, %38 {dfg_id = 66 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %41 = neura.reserve {dfg_id = 18 : i32} : !neura.data<!llvm.ptr, i1>
    %42 = "neura.data_mov"(%28) {dfg_id = 48 : i32, mapping_locs = [{id = 449 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 449 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %43 = neura.phi_start %42, %41 {dfg_id = 62 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 3 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
    %44 = neura.reserve {dfg_id = 19 : i32} : !neura.data<!llvm.ptr, i1>
    %45 = "neura.data_mov"(%31) {dfg_id = 52 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %46 = neura.phi_start %45, %44 {dfg_id = 64 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
    %47 = neura.reserve {dfg_id = 20 : i32} : !neura.data<i64, i1>
    %48 = "neura.data_mov"(%34) {dfg_id = 56 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %49 = neura.phi_start %48, %47 {dfg_id = 68 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 4 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %50 = neura.reserve {dfg_id = 21 : i32} : !neura.data<i64, i1>
    %51 = "neura.data_mov"(%10) {dfg_id = 59 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %52 = neura.phi_start %51, %50 {dfg_id = 71 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %53 = neura.reserve {dfg_id = 22 : i32} : !neura.data<i64, i1>
    %54 = "neura.data_mov"(%13) {dfg_id = 58 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %55 = neura.phi_start %54, %53 {dfg_id = 70 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %56 = neura.reserve {dfg_id = 23 : i32} : !neura.data<!llvm.ptr, i1>
    %57 = "neura.data_mov"(%16) {dfg_id = 50 : i32, mapping_locs = [{id = 256 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %58 = neura.phi_start %57, %56 {dfg_id = 63 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
    %59 = neura.reserve {dfg_id = 24 : i32} : !neura.data<!llvm.ptr, i1>
    %60 = "neura.data_mov"(%19) {dfg_id = 46 : i32, mapping_locs = [{id = 64 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %61 = neura.phi_start %60, %59 {dfg_id = 60 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
    %62 = neura.reserve {dfg_id = 25 : i32} : !neura.data<!llvm.ptr, i1>
    %63 = "neura.data_mov"(%22) {dfg_id = 47 : i32, mapping_locs = [{id = 416 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %64 = neura.phi_start %63, %62 {dfg_id = 61 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 1 : i32, y = 3 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<!llvm.ptr, i1> -> !neura.data<!llvm.ptr, i1>
    %65 = neura.reserve {dfg_id = 26 : i32} : !neura.data<i64, i1>
    %66 = "neura.data_mov"(%25) {dfg_id = 53 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 1 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %67 = neura.phi_start %66, %65 {dfg_id = 65 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 2 : i32, x = 0 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i64, i1> -> !neura.data<i64, i1>
    %68 = "neura.data_mov"(%64) {dfg_id = 77 : i32, mapping_locs = [{id = 40 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %69 = "neura.data_mov"(%67) {dfg_id = 88 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 26 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 384 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 384 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %70 = "neura.gep"(%68, %69) <{operandSegmentSizes = array<i32: 1, 1>}> {dfg_id = 104 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
    %71 = "neura.data_mov"(%70) {dfg_id = 110 : i32, mapping_locs = [{id = 384 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %72 = "neura.load"(%71) {dfg_id = 116 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %73 = "neura.data_mov"(%31) {dfg_id = 51 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %74 = "neura.data_mov"(%34) {dfg_id = 55 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 12 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 257 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 257 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %75 = neura.load_indexed %73[%74 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> {dfg_id = 67 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i32, i1>
    %76 = "neura.data_mov"(%61) {dfg_id = 74 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %77 = "neura.data_mov"(%49) {dfg_id = 94 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 32 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %78 = "neura.data_mov"(%67) {dfg_id = 87 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}, {id = 15 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 33 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}, {id = 33 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %79 = "neura.gep"(%76, %77, %78) <{operandSegmentSizes = array<i32: 1, 2>}> {dfg_id = 105 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>, !neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<!llvm.ptr, i1>
    %80 = "neura.data_mov"(%79) {dfg_id = 112 : i32, mapping_locs = [{id = 3 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %81 = "neura.load"(%80) {dfg_id = 118 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %82 = "neura.data_mov"(%81) {dfg_id = 122 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 2 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 1 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %83 = "neura.data_mov"(%75) {dfg_id = 91 : i32, mapping_locs = [{id = 25 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 128 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 128 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %84 = "neura.data_mov"(%72) {dfg_id = 120 : i32, mapping_locs = [{id = 39 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 25 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 130 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %85 = "neura.mul_add"(%82, %83, %84) {dfg_id = 134 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %86 = "neura.data_mov"(%85) {dfg_id = 146 : i32, mapping_locs = [{id = 12 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %87 = "neura.data_mov"(%70) {dfg_id = 109 : i32, mapping_locs = [{id = 39 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 257 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 257 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 257 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 257 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    "neura.store"(%86, %87) {dfg_id = 169 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 0 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    %88 = "neura.data_mov"(%37) {dfg_id = 96 : i32, mapping_locs = [{id = 44 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %89 = "neura.load"(%88) {dfg_id = 106 : i32, mapping_locs = [{id = 15 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 3 : i32, y = 3 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %90 = "neura.data_mov"(%79) {dfg_id = 111 : i32, mapping_locs = [{id = 32 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %91 = "neura.load"(%90) {dfg_id = 117 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 0 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<i32, i1>
    %92 = "neura.data_mov"(%58) {dfg_id = 82 : i32, mapping_locs = [{id = 256 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 256 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %93 = "neura.data_mov"(%67) {dfg_id = 86 : i32, mapping_locs = [{id = 129 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 2 : i32}, {id = 12 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 258 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 4 : i32}, {id = 258 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 258 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 258 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %94 = neura.load_indexed %92[%93 : !neura.data<i64, i1>] !neura.data<!llvm.ptr, i1> {dfg_id = 103 : i32, mapping_locs = [{id = 8 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 0 : i32, y = 2 : i32}]} : !neura.data<i32, i1>
    %95 = "neura.data_mov"(%94) {dfg_id = 108 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 288 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %96 = "neura.data_mov"(%91) {dfg_id = 121 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 16 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 289 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %97 = "neura.data_mov"(%89) {dfg_id = 113 : i32, mapping_locs = [{id = 46 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 43 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 42 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %98 = "neura.mul_add"(%95, %96, %97) {dfg_id = 133 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
    %99 = "neura.data_mov"(%98) {dfg_id = 145 : i32, mapping_locs = [{id = 288 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i32, i1>) -> !neura.data<i32, i1>
    %100 = "neura.data_mov"(%37) {dfg_id = 95 : i32, mapping_locs = [{id = 43 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 42 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 290 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 290 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 290 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    "neura.store"(%99, %100) {dfg_id = 168 : i32, mapping_locs = [{id = 9 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 2 : i32}]} : (!neura.data<i32, i1>, !neura.data<!llvm.ptr, i1>) -> ()
    %101 = "neura.data_mov"(%67) {dfg_id = 85 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %102 = "neura.data_mov"(%55) {dfg_id = 99 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 2 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %103 = "neura.add"(%101, %102) {dfg_id = 107 : i32, mapping_locs = [{id = 4 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 3 : i32, x = 0 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %104 = "neura.data_mov"(%103) {dfg_id = 115 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 14 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %105 = "neura.data_mov"(%52) {dfg_id = 102 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 3 : i32}, {id = 192 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %106 = "neura.icmp"(%104, %105) <{cmpType = "eq"}> {dfg_id = 119 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 5 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
    %107 = "neura.data_mov"(%106) {dfg_id = 132 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %108 = "neura.not"(%107) {dfg_id = 144 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %109 = "neura.data_mov"(%103) {dfg_id = 114 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 10 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 4 : i32}, {id = 163 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 5 : i32}, {id = 163 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 6 : i32}, {id = 163 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %110 = "neura.data_mov"(%108) {dfg_id = 167 : i32, mapping_locs = [{id = 194 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}, {id = 17 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %111 = neura.grant_predicate %109, %110 {dfg_id = 180 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %111 -> %65 {dfg_id = 192 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 129 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 129 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 129 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 129 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 129 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 129 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 129 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %112 = "neura.data_mov"(%64) {dfg_id = 76 : i32, mapping_locs = [{id = 41 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 45 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 323 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 7 : i32}, {id = 323 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 8 : i32}, {id = 323 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 9 : i32}, {id = 323 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 10 : i32}, {id = 323 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %113 = "neura.data_mov"(%108) {dfg_id = 166 : i32, mapping_locs = [{id = 203 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 11 : i32, resource = "register", time_step = 6 : i32}, {id = 201 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 7 : i32}, {id = 197 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 8 : i32}, {id = 18 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 21 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 20 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %114 = neura.grant_predicate %112, %113 {dfg_id = 179 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %114 -> %62 {dfg_id = 191 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 30 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 417 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 417 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 417 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 417 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 417 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %115 = "neura.data_mov"(%61) {dfg_id = 73 : i32, mapping_locs = [{id = 67 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 5 : i32}, {id = 67 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 6 : i32}, {id = 67 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 7 : i32}, {id = 67 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 8 : i32}, {id = 67 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %116 = "neura.data_mov"(%108) {dfg_id = 165 : i32, mapping_locs = [{id = 202 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 6 : i32}, {id = 200 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 7 : i32}, {id = 196 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 8 : i32}, {id = 19 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %117 = neura.grant_predicate %115, %116 {dfg_id = 178 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %117 -> %59 {dfg_id = 190 : i32, mapping_locs = [{id = 65 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 65 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 65 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 65 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 65 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 65 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 65 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 65 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 65 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %118 = "neura.data_mov"(%58) {dfg_id = 81 : i32, mapping_locs = [{id = 259 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 6 : i32}, {id = 24 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 28 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 322 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 322 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %119 = "neura.data_mov"(%108) {dfg_id = 164 : i32, mapping_locs = [{id = 201 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 6 : i32}, {id = 199 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 7 : i32}, {id = 195 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 8 : i32}, {id = 195 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 9 : i32}, {id = 20 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %120 = neura.grant_predicate %118, %119 {dfg_id = 177 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %120 -> %56 {dfg_id = 189 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 27 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 259 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 13 : i32}, {id = 259 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 14 : i32}, {id = 259 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 15 : i32}, {id = 259 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 16 : i32}, {id = 259 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 17 : i32}, {id = 259 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 18 : i32}, {id = 259 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 19 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %121 = "neura.data_mov"(%55) {dfg_id = 98 : i32, mapping_locs = [{id = 162 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 2 : i32}, {id = 162 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 3 : i32}, {id = 162 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 4 : i32}, {id = 162 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 162 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %122 = "neura.data_mov"(%108) {dfg_id = 163 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %123 = neura.grant_predicate %121, %122 {dfg_id = 176 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %123 -> %53 {dfg_id = 188 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 160 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 160 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 160 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 160 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}, {id = 160 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 160 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}, {id = 160 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}, {id = 160 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %124 = "neura.data_mov"(%52) {dfg_id = 101 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}, {id = 321 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}, {id = 321 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 321 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %125 = "neura.data_mov"(%108) {dfg_id = 162 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %126 = neura.grant_predicate %124, %125 {dfg_id = 175 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %126 -> %50 {dfg_id = 187 : i32, mapping_locs = [{id = 321 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 321 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 321 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 321 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 321 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 321 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 321 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 321 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 321 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 321 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %127 = "neura.data_mov"(%49) {dfg_id = 93 : i32, mapping_locs = [{id = 164 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 4 : i32}, {id = 164 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 5 : i32}, {id = 164 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 6 : i32}, {id = 164 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 7 : i32}, {id = 164 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 8 : i32}, {id = 164 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 9 : i32}, {id = 164 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 10 : i32}, {id = 164 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %128 = "neura.data_mov"(%108) {dfg_id = 161 : i32, mapping_locs = [{id = 200 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 6 : i32}, {id = 198 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 7 : i32}, {id = 19 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 5 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 4 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 161 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %129 = neura.grant_predicate %127, %128 {dfg_id = 174 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %129 -> %47 {dfg_id = 186 : i32, mapping_locs = [{id = 163 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 12 : i32}, {id = 163 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 13 : i32}, {id = 163 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 14 : i32}, {id = 163 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 15 : i32}, {id = 163 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 16 : i32}, {id = 163 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 17 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %130 = "neura.data_mov"(%46) {dfg_id = 84 : i32, mapping_locs = [{id = 128 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 10 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 161 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 161 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %131 = "neura.data_mov"(%108) {dfg_id = 160 : i32, mapping_locs = [{id = 199 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 6 : i32}, {id = 197 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 7 : i32}, {id = 18 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 21 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 17 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %132 = neura.grant_predicate %130, %131 {dfg_id = 173 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %132 -> %44 {dfg_id = 185 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 130 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 130 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 130 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}, {id = 130 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 15 : i32}, {id = 130 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 16 : i32}, {id = 130 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 17 : i32}, {id = 130 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 18 : i32}, {id = 130 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 19 : i32}, {id = 130 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 20 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %133 = "neura.data_mov"(%43) {dfg_id = 79 : i32, mapping_locs = [{id = 448 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 45 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 320 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %134 = "neura.data_mov"(%108) {dfg_id = 159 : i32, mapping_locs = [{id = 198 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 6 : i32}, {id = 196 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 7 : i32}, {id = 193 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}, {id = 20 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %135 = neura.grant_predicate %133, %134 {dfg_id = 172 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %135 -> %41 {dfg_id = 184 : i32, mapping_locs = [{id = 34 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 450 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 450 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 450 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 450 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}, {id = 450 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 15 : i32}, {id = 450 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 16 : i32}, {id = 450 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 17 : i32}, {id = 450 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 18 : i32}, {id = 450 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 19 : i32}, {id = 450 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 20 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %136 = "neura.data_mov"(%40) {dfg_id = 90 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}, {id = 0 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %137 = "neura.data_mov"(%108) {dfg_id = 158 : i32, mapping_locs = [{id = 197 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 6 : i32}, {id = 19 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 5 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %138 = neura.grant_predicate %136, %137 {dfg_id = 171 : i32, mapping_locs = [{id = 1 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %138 -> %38 {dfg_id = 183 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 1 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}, {id = 1 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 1 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}, {id = 1 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}, {id = 1 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 14 : i32}, {id = 1 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 1 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 1 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 1 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}, {id = 1 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 19 : i32}, {id = 1 : i32, index_per_ii = 6 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 20 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %139 = "neura.data_mov"(%49) {dfg_id = 92 : i32, mapping_locs = [{id = 160 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 14 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 192 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %140 = "neura.data_mov"(%106) {dfg_id = 131 : i32, mapping_locs = [{id = 193 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}, {id = 193 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %141 = neura.grant_predicate %139, %140 {dfg_id = 143 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 7 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %142 = "neura.data_mov"(%55) {dfg_id = 97 : i32, mapping_locs = [{id = 161 : i32, index_per_ii = 2 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 2 : i32}, {id = 161 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 3 : i32}, {id = 161 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 4 : i32}, {id = 161 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %143 = "neura.data_mov"(%106) {dfg_id = 130 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %144 = neura.grant_predicate %142, %143 {dfg_id = 142 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %145 = "neura.data_mov"(%52) {dfg_id = 100 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 3 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 3 : i32}, {id = 320 : i32, index_per_ii = 4 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 4 : i32}, {id = 320 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %146 = "neura.data_mov"(%106) {dfg_id = 129 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %147 = neura.grant_predicate %145, %146 {dfg_id = 141 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 6 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %148 = "neura.data_mov"(%46) {dfg_id = 83 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 162 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}, {id = 162 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %149 = "neura.data_mov"(%106) {dfg_id = 128 : i32, mapping_locs = [{id = 197 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 5 : i32}, {id = 196 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 6 : i32}, {id = 18 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 21 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}, {id = 17 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %150 = neura.grant_predicate %148, %149 {dfg_id = 140 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %151 = "neura.data_mov"(%43) {dfg_id = 78 : i32, mapping_locs = [{id = 45 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 320 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %152 = "neura.data_mov"(%106) {dfg_id = 127 : i32, mapping_locs = [{id = 196 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 5 : i32}, {id = 195 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 6 : i32}, {id = 195 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 7 : i32}, {id = 20 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %153 = neura.grant_predicate %151, %152 {dfg_id = 139 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %154 = "neura.data_mov"(%40) {dfg_id = 89 : i32, mapping_locs = [{id = 0 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 3 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %155 = "neura.data_mov"(%106) {dfg_id = 126 : i32, mapping_locs = [{id = 195 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 5 : i32}, {id = 19 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 66 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}, {id = 66 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %156 = neura.grant_predicate %154, %155 {dfg_id = 138 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    %157 = "neura.data_mov"(%64) {dfg_id = 75 : i32, mapping_locs = [{id = 42 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 29 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 161 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}, {id = 161 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %158 = "neura.data_mov"(%106) {dfg_id = 125 : i32, mapping_locs = [{id = 194 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 5 : i32}, {id = 18 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 21 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 17 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 8 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %159 = neura.grant_predicate %157, %158 {dfg_id = 137 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %160 = "neura.data_mov"(%61) {dfg_id = 72 : i32, mapping_locs = [{id = 64 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 5 : i32}, {id = 64 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 64 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %161 = "neura.data_mov"(%106) {dfg_id = 124 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 65 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 6 : i32}, {id = 65 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %162 = neura.grant_predicate %160, %161 {dfg_id = 136 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %163 = "neura.data_mov"(%58) {dfg_id = 80 : i32, mapping_locs = [{id = 24 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 28 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %164 = "neura.data_mov"(%106) {dfg_id = 123 : i32, mapping_locs = [{id = 18 : i32, index_per_ii = 5 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 5 : i32}, {id = 21 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 20 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %165 = neura.grant_predicate %163, %164 {dfg_id = 135 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    %166 = "neura.data_mov"(%141) {dfg_id = 157 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %167 = "neura.data_mov"(%144) {dfg_id = 156 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 193 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 7 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %168 = "neura.add"(%166, %167) {dfg_id = 170 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 8 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i64, i1>
    %169 = "neura.data_mov"(%168) {dfg_id = 182 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %170 = "neura.data_mov"(%147) {dfg_id = 154 : i32, mapping_locs = [{id = 33 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 6 : i32}, {id = 194 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 7 : i32}, {id = 194 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 8 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %171 = "neura.icmp"(%169, %170) <{cmpType = "eq"}> {dfg_id = 193 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 9 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i64, i1>, !neura.data<i64, i1>) -> !neura.data<i1, i1>
    %172 = "neura.data_mov"(%171) {dfg_id = 196 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %173 = "neura.not"(%172) {dfg_id = 198 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 10 : i32, x = 2 : i32, y = 1 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %174 = "neura.data_mov"(%168) {dfg_id = 181 : i32, mapping_locs = [{id = 202 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 8 : i32}, {id = 202 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 9 : i32}, {id = 202 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 10 : i32}, {id = 202 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 11 : i32}, {id = 202 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 12 : i32}, {id = 202 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 10 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %175 = "neura.data_mov"(%173) {dfg_id = 208 : i32, mapping_locs = [{id = 203 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 11 : i32, resource = "register", time_step = 10 : i32}, {id = 203 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 11 : i32, resource = "register", time_step = 11 : i32}, {id = 203 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 11 : i32, resource = "register", time_step = 12 : i32}, {id = 203 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 11 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %176 = neura.grant_predicate %174, %175 {dfg_id = 218 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 14 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %176 -> %32 {dfg_id = 227 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}, {id = 164 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 15 : i32}, {id = 164 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 16 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %177 = "neura.data_mov"(%150) {dfg_id = 152 : i32, mapping_locs = [{id = 162 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}, {id = 162 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 162 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 12 : i32}, {id = 162 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %178 = "neura.data_mov"(%173) {dfg_id = 207 : i32, mapping_locs = [{id = 201 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 9 : i32, resource = "register", time_step = 10 : i32}, {id = 193 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 11 : i32}, {id = 17 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 161 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %179 = neura.grant_predicate %177, %178 {dfg_id = 217 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 14 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %179 -> %29 {dfg_id = 226 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}, {id = 131 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 15 : i32}, {id = 131 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 16 : i32}, {id = 131 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 17 : i32}, {id = 131 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 18 : i32}, {id = 131 : i32, index_per_ii = 5 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 19 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %180 = "neura.data_mov"(%153) {dfg_id = 151 : i32, mapping_locs = [{id = 325 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 9 : i32}, {id = 325 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 10 : i32}, {id = 325 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 11 : i32}, {id = 325 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 12 : i32}, {id = 325 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 13 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %181 = "neura.data_mov"(%173) {dfg_id = 206 : i32, mapping_locs = [{id = 200 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 8 : i32, resource = "register", time_step = 10 : i32}, {id = 18 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 21 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 20 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %182 = neura.grant_predicate %180, %181 {dfg_id = 216 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "tile", time_step = 14 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %182 -> %26 {dfg_id = 225 : i32, mapping_locs = [{id = 34 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}, {id = 449 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 15 : i32}, {id = 449 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 16 : i32}, {id = 449 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 17 : i32}, {id = 449 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 18 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %183 = "neura.data_mov"(%156) {dfg_id = 150 : i32, mapping_locs = [{id = 66 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 66 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}, {id = 66 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %184 = "neura.data_mov"(%173) {dfg_id = 205 : i32, mapping_locs = [{id = 199 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 7 : i32, resource = "register", time_step = 10 : i32}, {id = 19 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %185 = neura.grant_predicate %183, %184 {dfg_id = 215 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %185 -> %23 {dfg_id = 224 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 2 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 2 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %186 = "neura.data_mov"(%159) {dfg_id = 149 : i32, mapping_locs = [{id = 14 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 9 : i32}, {id = 196 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 10 : i32}, {id = 196 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 11 : i32}, {id = 196 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %187 = "neura.data_mov"(%173) {dfg_id = 204 : i32, mapping_locs = [{id = 197 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 10 : i32}, {id = 197 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 11 : i32}, {id = 197 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %188 = neura.grant_predicate %186, %187 {dfg_id = 214 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 13 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %188 -> %20 {dfg_id = 223 : i32, mapping_locs = [{id = 17 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 16 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}, {id = 30 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 15 : i32}, {id = 416 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 16 : i32}, {id = 416 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 17 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %189 = "neura.data_mov"(%162) {dfg_id = 148 : i32, mapping_locs = [{id = 64 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 8 : i32}, {id = 64 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 9 : i32}, {id = 64 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %190 = "neura.data_mov"(%173) {dfg_id = 203 : i32, mapping_locs = [{id = 19 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %191 = neura.grant_predicate %189, %190 {dfg_id = 213 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 0 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %191 -> %17 {dfg_id = 222 : i32, mapping_locs = [{id = 64 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}, {id = 64 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 12 : i32}, {id = 64 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 13 : i32}, {id = 64 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 14 : i32}, {id = 64 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}, {id = 64 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 16 : i32}, {id = 64 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 17 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %192 = "neura.data_mov"(%165) {dfg_id = 147 : i32, mapping_locs = [{id = 324 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 8 : i32}, {id = 324 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 9 : i32}, {id = 324 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 10 : i32}, {id = 324 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 11 : i32}, {id = 324 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 4 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<!llvm.ptr, i1>) -> !neura.data<!llvm.ptr, i1>
    %193 = "neura.data_mov"(%173) {dfg_id = 202 : i32, mapping_locs = [{id = 18 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 10 : i32}, {id = 23 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 35 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %194 = neura.grant_predicate %192, %193 {dfg_id = 212 : i32, mapping_locs = [{id = 10 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 13 : i32, x = 2 : i32, y = 2 : i32}]} : !neura.data<!llvm.ptr, i1>, !neura.data<i1, i1> -> !neura.data<!llvm.ptr, i1>
    neura.ctrl_mov %194 -> %14 {dfg_id = 221 : i32, mapping_locs = [{id = 31 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 27 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, resource = "link", time_step = 14 : i32}, {id = 256 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 15 : i32}, {id = 256 : i32, index_per_ii = 2 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 16 : i32}, {id = 256 : i32, index_per_ii = 3 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 17 : i32}, {id = 256 : i32, index_per_ii = 4 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 18 : i32}]} : !neura.data<!llvm.ptr, i1> !neura.data<!llvm.ptr, i1>
    %195 = "neura.data_mov"(%144) {dfg_id = 155 : i32, mapping_locs = [{id = 165 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 6 : i32}, {id = 165 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 7 : i32}, {id = 165 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 8 : i32}, {id = 165 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 9 : i32}, {id = 165 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 10 : i32}, {id = 165 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 11 : i32}, {id = 165 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 5 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %196 = "neura.data_mov"(%173) {dfg_id = 201 : i32, mapping_locs = [{id = 195 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 10 : i32}, {id = 17 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 11 : i32}, {id = 161 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 12 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %197 = neura.grant_predicate %195, %196 {dfg_id = 211 : i32, mapping_locs = [{id = 5 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 13 : i32, x = 1 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %197 -> %11 {dfg_id = 220 : i32, mapping_locs = [{id = 13 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 13 : i32}, {id = 131 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 3 : i32, resource = "register", time_step = 14 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %198 = "neura.data_mov"(%147) {dfg_id = 153 : i32, mapping_locs = [{id = 320 : i32, index_per_ii = 6 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 6 : i32}, {id = 33 : i32, index_per_ii = 7 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 7 : i32}, {id = 198 : i32, index_per_ii = 8 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 8 : i32}, {id = 198 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 9 : i32}, {id = 198 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 10 : i32}, {id = 198 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 6 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i64, i1>) -> !neura.data<i64, i1>
    %199 = "neura.data_mov"(%173) {dfg_id = 200 : i32, mapping_locs = [{id = 192 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 10 : i32}, {id = 192 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 0 : i32, resource = "register", time_step = 11 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %200 = neura.grant_predicate %198, %199 {dfg_id = 210 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 12 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i64, i1>, !neura.data<i1, i1> -> !neura.data<i64, i1>
    neura.ctrl_mov %200 -> %8 {dfg_id = 219 : i32, mapping_locs = [{id = 20 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}, {id = 322 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 13 : i32}, {id = 322 : i32, index_per_ii = 0 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 14 : i32}, {id = 322 : i32, index_per_ii = 1 : i32, invalid_iterations = 1 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 15 : i32}]} : !neura.data<i64, i1> !neura.data<i64, i1>
    %201 = "neura.data_mov"(%171) {dfg_id = 194 : i32, mapping_locs = [{id = 193 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 9 : i32}, {id = 193 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 1 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %202 = "neura.data_mov"(%171) {dfg_id = 195 : i32, mapping_locs = [{id = 194 : i32, index_per_ii = 9 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 9 : i32}, {id = 194 : i32, index_per_ii = 10 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 10 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    %203 = neura.grant_predicate %201, %202 {dfg_id = 197 : i32, mapping_locs = [{id = 6 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 11 : i32, x = 2 : i32, y = 1 : i32}]} : !neura.data<i1, i1>, !neura.data<i1, i1> -> !neura.data<i1, i1>
    %204 = "neura.data_mov"(%203) {dfg_id = 199 : i32, mapping_locs = [{id = 194 : i32, index_per_ii = 11 : i32, invalid_iterations = 0 : i32, per_tile_register_id = 2 : i32, resource = "register", time_step = 11 : i32}, {id = 19 : i32, index_per_ii = 12 : i32, invalid_iterations = 0 : i32, resource = "link", time_step = 12 : i32}]} : (!neura.data<i1, i1>) -> !neura.data<i1, i1>
    neura.return_void %204 : !neura.data<i1, i1> {dfg_id = 209 : i32, mapping_locs = [{id = 2 : i32, index_per_ii = 13 : i32, invalid_iterations = 0 : i32, resource = "tile", time_step = 13 : i32, x = 2 : i32, y = 0 : i32}]}
    neura.yield {dfg_id = 27 : i32}
  }
  llvm.func local_unnamed_addr @main() -> (i32 {llvm.noundef}) attributes {memory_effects = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = none>, no_unwind, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic"} {
    %0 = llvm.mlir.addressof @p : !llvm.ptr
    %1 = llvm.mlir.addressof @A : !llvm.ptr
    %2 = llvm.mlir.addressof @s : !llvm.ptr
    %3 = llvm.mlir.addressof @q : !llvm.ptr
    %4 = llvm.mlir.addressof @r : !llvm.ptr
    %5 = "neura.constant"() <{value = 0 : i64}> : () -> i64
    %6 = "neura.constant"() <{value = 0 : i32}> : () -> i32
    %7 = "neura.data_mov"(%5) : (i64) -> i64
    neura.br %7 : i64 to ^bb1
  ^bb1(%8: i64):  // 2 preds: ^bb0, ^bb3
    %9 = "neura.data_mov"(%3) : (!llvm.ptr) -> !llvm.ptr
    %10 = "neura.data_mov"(%8) : (i64) -> i64
    %11 = "neura.gep"(%9, %10) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
    %12 = "neura.data_mov"(%4) : (!llvm.ptr) -> !llvm.ptr
    %13 = "neura.data_mov"(%8) : (i64) -> i64
    %14 = neura.load_indexed %12[%13 : i64] !llvm.ptr : i32
    %15 = "neura.data_mov"(%11) : (!llvm.ptr) -> !llvm.ptr
    %16 = "neura.load"(%15) : (!llvm.ptr) -> i32
    %17 = "neura.data_mov"(%16) : (i32) -> i32
    %18 = "neura.data_mov"(%5) : (i64) -> i64
    neura.br %17, %18 : i32, i64 to ^bb2
  ^bb2(%19: i32, %20: i64):  // 2 preds: ^bb1, ^bb2
    %21 = "neura.data_mov"(%2) : (!llvm.ptr) -> !llvm.ptr
    %22 = "neura.data_mov"(%20) : (i64) -> i64
    %23 = "neura.gep"(%21, %22) <{operandSegmentSizes = array<i32: 1, 1>}> : (!llvm.ptr, i64) -> !llvm.ptr
    %24 = "neura.data_mov"(%23) : (!llvm.ptr) -> !llvm.ptr
    %25 = "neura.load"(%24) : (!llvm.ptr) -> i32
    %26 = "neura.data_mov"(%1) : (!llvm.ptr) -> !llvm.ptr
    %27 = "neura.data_mov"(%8) : (i64) -> i64
    %28 = "neura.data_mov"(%20) : (i64) -> i64
    %29 = neura.load_indexed %26[%27, %28 : i64, i64] !llvm.ptr : i32
    %30 = "neura.data_mov"(%29) : (i32) -> i32
    %31 = "neura.data_mov"(%14) : (i32) -> i32
    %32 = "neura.data_mov"(%25) : (i32) -> i32
    %33 = "neura.mul_add"(%30, %31, %32) : (i32, i32, i32) -> i32
    %34 = "neura.data_mov"(%33) : (i32) -> i32
    %35 = "neura.data_mov"(%23) : (!llvm.ptr) -> !llvm.ptr
    "neura.store"(%34, %35) : (i32, !llvm.ptr) -> ()
    %36 = "neura.data_mov"(%0) : (!llvm.ptr) -> !llvm.ptr
    %37 = "neura.data_mov"(%20) : (i64) -> i64
    %38 = neura.load_indexed %36[%37 : i64] !llvm.ptr : i32
    %39 = "neura.data_mov"(%38) : (i32) -> i32
    %40 = "neura.data_mov"(%29) : (i32) -> i32
    %41 = "neura.data_mov"(%19) : (i32) -> i32
    %42 = "neura.mul_add"(%39, %40, %41) : (i32, i32, i32) -> i32
    %43 = "neura.data_mov"(%20) : (i64) -> i64
    %44 = "neura.add"(%43) {rhs_value = 1 : i64} : (i64) -> i64
    %45 = "neura.data_mov"(%44) : (i64) -> i64
    %46 = "neura.icmp"(%45) <{cmpType = "eq"}> {rhs_value = 1024 : i64} : (i64) -> i1
    %47 = "neura.data_mov"(%46) : (i1) -> i1
    %48 = "neura.data_mov"(%42) : (i32) -> i32
    %49 = "neura.data_mov"(%44) : (i64) -> i64
    neura.cond_br %47 : i1 then to ^bb3 else %48, %49 : i32, i64 to ^bb2
  ^bb3:  // pred: ^bb2
    %50 = "neura.data_mov"(%42) : (i32) -> i32
    %51 = "neura.data_mov"(%11) : (!llvm.ptr) -> !llvm.ptr
    "neura.store"(%50, %51) : (i32, !llvm.ptr) -> ()
    %52 = "neura.data_mov"(%8) : (i64) -> i64
    %53 = "neura.add"(%52) {rhs_value = 1 : i64} : (i64) -> i64
    %54 = "neura.data_mov"(%53) : (i64) -> i64
    %55 = "neura.icmp"(%54) <{cmpType = "eq"}> {rhs_value = 1024 : i64} : (i64) -> i1
    %56 = "neura.data_mov"(%55) : (i1) -> i1
    %57 = "neura.data_mov"(%53) : (i64) -> i64
    neura.cond_br %56 : i1 then to ^bb4 else %57 : i64 to ^bb1
  ^bb4:  // pred: ^bb3
    %58 = "neura.data_mov"(%6) : (i32) -> i32
    "neura.return"(%58) : (i32) -> ()
  }
}

